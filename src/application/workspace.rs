//! Workspaces — physically separate memories (PLAN §2, PHASE2-DESIGN §2).
//!
//! A workspace is a directory `<home>/workspaces/<name>/` holding its own STM
//! and LTM store files. Nothing is shared between workspaces: no index, no
//! session buffer, no query path. One workspace is *active* per process; the
//! MCP `workspace_switch` tool swaps it (rebuilding every service, which also
//! resets the session buffers).
//!
//! This module holds the policy (names, confirmation, which workspace may be
//! deleted, switching) and the switchable handle. Storage lives behind the
//! [`WorkspaceHost`] port, implemented by the composition root.

use crate::application::app::NeurolitheApp;
use crate::application::documents::DocumentService;
use crate::application::introspection::IntrospectionService;
use crate::application::query_service::QueryService;
use anyhow::{Result, anyhow, bail};
use serde::Serialize;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

/// Longest workspace name (the regex `^[a-z0-9][a-z0-9_-]{0,63}$`).
pub const MAX_NAME_LEN: usize = 64;

/// Validate a workspace name against `^[a-z0-9][a-z0-9_-]{0,63}$`. The rule
/// also guarantees the name is a safe single path component (no `/`, `..`,
/// dots, or uppercase/case-folding surprises).
pub fn validate_name(name: &str) -> Result<()> {
    let bytes = name.as_bytes();
    let first_ok = bytes
        .first()
        .is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit());
    let rest_ok = bytes
        .iter()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || *b == b'_' || *b == b'-');
    if first_ok && rest_ok && bytes.len() <= MAX_NAME_LEN {
        Ok(())
    } else {
        bail!(
            "invalid workspace name {name:?}: use 1-{MAX_NAME_LEN} characters from a-z, 0-9, \
             '_' and '-', starting with a letter or digit"
        )
    }
}

/// One workspace, as reported by `workspace_current` / `workspace_list`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WorkspaceInfo {
    pub name: String,
    pub path: String,
    pub stm_bytes: u64,
    pub ltm_bytes: u64,
    pub active: bool,
}

/// The use-case services bound to one workspace's stores.
pub struct WorkspaceServices {
    pub app: Arc<NeurolitheApp>,
    pub introspection: Arc<IntrospectionService>,
    pub query: QueryService,
    /// LTM document filing (`remember_document`).
    pub documents: DocumentService,
}

/// An opened workspace: its name plus the services over its stores.
pub struct OpenWorkspace {
    pub name: String,
    pub services: WorkspaceServices,
}

/// Storage-side operations on workspaces (implemented by the composition root
/// over the SQLite layout). All names passed in are already validated.
#[async_trait::async_trait(?Send)]
pub trait WorkspaceHost {
    /// Open (creating on demand) a workspace and build its services.
    async fn open(&self, name: &str) -> Result<WorkspaceServices>;
    /// Whether the workspace directory exists.
    fn exists(&self, name: &str) -> bool;
    /// Every workspace on disk (the `active` flag is left `false`).
    fn list(&self) -> Result<Vec<WorkspaceInfo>>;
    /// Create an empty workspace; `Ok(false)` if it already existed.
    fn create(&self, name: &str) -> Result<bool>;
    /// Remove a workspace and all of its data.
    fn delete(&self, name: &str) -> Result<()>;
    /// JSON dump of a workspace's STM facts + LTM leaves (read-only).
    fn export(&self, name: &str) -> Result<serde_json::Value>;
}

/// Owns the active workspace and applies the workspace rules. Shared (via
/// `Rc`) by the MCP server and the background tasks, so a switch is seen by
/// everyone on their next use.
pub struct WorkspaceManager {
    host: Box<dyn WorkspaceHost>,
    active: RefCell<Rc<OpenWorkspace>>,
    allow_switch: bool,
}

impl WorkspaceManager {
    /// Validate and open `name` (created on demand) as the active workspace.
    pub async fn start(
        host: Box<dyn WorkspaceHost>,
        name: &str,
        allow_switch: bool,
    ) -> Result<Self> {
        validate_name(name)?;
        let services = host.open(name).await?;
        Ok(Self::with_active(
            host,
            OpenWorkspace {
                name: name.to_string(),
                services,
            },
            allow_switch,
        ))
    }

    /// Start from an already-opened workspace (the daemon shares its stores
    /// with the Kafka loops and hands the same services to MCP).
    pub fn with_active(
        host: Box<dyn WorkspaceHost>,
        active: OpenWorkspace,
        allow_switch: bool,
    ) -> Self {
        Self {
            host,
            active: RefCell::new(Rc::new(active)),
            allow_switch,
        }
    }

    /// The active workspace. Hold the `Rc` for the duration of one operation.
    pub fn current(&self) -> Rc<OpenWorkspace> {
        self.active.borrow().clone()
    }

    pub fn current_name(&self) -> String {
        self.active.borrow().name.clone()
    }

    fn info_for(&self, name: &str) -> Result<WorkspaceInfo> {
        self.list()?
            .into_iter()
            .find(|w| w.name == name)
            .ok_or_else(|| anyhow!("workspace {name:?} does not exist"))
    }

    /// `workspace_current`.
    pub fn current_info(&self) -> Result<WorkspaceInfo> {
        self.info_for(&self.current_name())
    }

    /// `workspace_list`, with the active one flagged.
    pub fn list(&self) -> Result<Vec<WorkspaceInfo>> {
        let active = self.current_name();
        let mut all = self.host.list()?;
        for w in &mut all {
            w.active = w.name == active;
        }
        all.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(all)
    }

    /// `workspace_create`. Returns the new workspace's info.
    pub fn create(&self, name: &str) -> Result<WorkspaceInfo> {
        self.require_unpinned("create other workspaces")?;
        validate_name(name)?;
        if !self.host.create(name)? {
            bail!("workspace {name:?} already exists");
        }
        self.info_for(name)
    }

    /// `workspace_switch`: reopen the stores of an existing workspace and make
    /// it active. Every service is rebuilt, so session buffers start empty.
    pub async fn switch(&self, name: &str) -> Result<WorkspaceInfo> {
        if !self.allow_switch {
            bail!(
                "workspace switching is disabled ([mcp] allow_workspace_switch = false); \
                 this server is pinned to workspace {:?}",
                self.current_name()
            );
        }
        validate_name(name)?;
        if !self.host.exists(name) {
            bail!("workspace {name:?} does not exist; create it first with workspace_create");
        }
        let services = self.host.open(name).await?;
        *self.active.borrow_mut() = Rc::new(OpenWorkspace {
            name: name.to_string(),
            services,
        });
        self.info_for(name)
    }

    /// `workspace_delete`: `confirm` must repeat the name, and the active
    /// workspace cannot be deleted (switch away first).
    pub fn delete(&self, name: &str, confirm: Option<&str>) -> Result<()> {
        self.require_unpinned("delete other workspaces")?;
        validate_name(name)?;
        if confirm != Some(name) {
            bail!(
                "workspace_delete is destructive: pass confirm = {name:?} (the same value as \
                 name) to proceed"
            );
        }
        if name == self.current_name() {
            bail!("workspace {name:?} is active; switch to another workspace before deleting it");
        }
        if !self.host.exists(name) {
            bail!("workspace {name:?} does not exist");
        }
        self.host.delete(name)
    }

    /// `workspace_export`: the named workspace, or the active one.
    pub fn export(&self, name: Option<&str>) -> Result<serde_json::Value> {
        let name = name
            .map(str::to_string)
            .unwrap_or_else(|| self.current_name());
        validate_name(&name)?;
        if name != self.current_name() {
            self.require_unpinned("export other workspaces")?;
        }
        if !self.host.exists(&name) {
            bail!("workspace {name:?} does not exist");
        }
        self.host.export(&name)
    }

    /// With switching disabled, a session is pinned to its workspace and may
    /// not reach any other one — not by switching, nor by exporting, creating
    /// or deleting (P2R-4).
    fn require_unpinned(&self, action: &str) -> Result<()> {
        if self.allow_switch {
            Ok(())
        } else {
            bail!(
                "this server is pinned to workspace {:?} ([mcp] allow_workspace_switch = false); \
                 it cannot {action}",
                self.current_name()
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_name() {
        for ok in [
            "default",
            "a",
            "work-2",
            "novel_research",
            "0abc",
            &"a".repeat(64),
        ] {
            assert!(validate_name(ok).is_ok(), "{ok}");
        }
        for bad in [
            "",
            "-lead",
            "_lead",
            "Upper",
            "has space",
            "dot.name",
            "../escape",
            "a/b",
            "ünï",
            &"a".repeat(65),
        ] {
            assert!(validate_name(bad).is_err(), "{bad:?} should be rejected");
        }
    }
}
