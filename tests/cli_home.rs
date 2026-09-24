//! Phase 2 §1: home/config resolution, `.env` handling, file permissions, and
//! `neurolithe init`. The CWD must never be consulted (SEC-03).
mod common;

use common::{API_KEY, FakeLlm, Home, McpProcess, bin, home_config_toml, run_cli};
use serde_json::{Value, json};
use std::path::Path;

fn spawn_raw(home: &Home, args: &[&str], env: &[(&str, &str)]) -> McpProcess {
    McpProcess::spawn_with(
        Path::new(bin()),
        args,
        home.cwd.path(),
        home.cwd.path(),
        env,
    )
}

/// Authorization headers the fake LLM saw.
fn auths(llm: &FakeLlm) -> Vec<String> {
    llm.requests()
        .iter()
        .filter_map(|r| r.authorization.clone())
        .collect()
}

/// SEC-03: a `neurolithe.toml` and `.env` in the CWD (e.g. a cloned repo the
/// MCP client happens to start in) must be IGNORED. Config and `.env` come
/// only from the home.
#[test]
fn sec03_cwd_config_and_dotenv_are_ignored() {
    let good = FakeLlm::start();
    let evil = FakeLlm::start();
    let home = Home::new(&good.base_url());
    std::fs::write(home.path().join(".env"), "OPENAI_API_KEY=home-key\n").unwrap();

    // A hostile CWD: config pointing at another endpoint + store paths +
    // a workspace, and a .env with another key and workspace.
    let cwd = home.cwd.path();
    std::fs::write(
        cwd.join("neurolithe.toml"),
        format!(
            "workspace = \"cwd-ws\"\n{}\n",
            home_config_toml(&evil.base_url())
                + &format!("\n[stm]\npath = \"{}/cwd-stm.sqlite\"\n", cwd.display())
        ),
    )
    .unwrap();
    std::fs::write(
        cwd.join(".env"),
        "OPENAI_API_KEY=cwd-key\nNEUROLITHE_API_KEY=cwd-key\nNEUROLITHE_WORKSPACE=cwd-env-ws\n",
    )
    .unwrap();
    let cwd_before: Vec<_> = std::fs::read_dir(cwd)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();

    // No key in the process env: the only valid key source is <home>/.env.
    let home_str = home.path().display().to_string();
    let mut server = spawn_raw(&home, &["mcp"], &[("NEUROLITHE_HOME", home_str.as_str())]);
    server.initialize();
    let cur = server.call_tool("workspace_current", json!({}));
    assert!(!cur.is_error, "{}", cur.text);
    assert_eq!(
        cur.json()["name"],
        "default",
        "SEC-03: CWD selected the workspace"
    );

    let res = server.call_tool("store_memory", json!({"fact_text": "where do I go?"}));
    assert!(!res.is_error, "{} / stderr: {}", res.text, server.stderr());

    assert!(
        evil.requests().is_empty(),
        "SEC-03: CWD config's endpoint was used"
    );
    assert!(!good.requests().is_empty(), "home config's endpoint unused");
    assert!(
        auths(&good).iter().all(|a| a == "Bearer home-key"),
        "SEC-03: key did not come from <home>/.env: {:?}",
        auths(&good)
    );
    let cwd_after: Vec<_> = std::fs::read_dir(cwd)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(
        cwd_before.len(),
        cwd_after.len(),
        "SEC-03: files created in CWD: {cwd_after:?}"
    );
    assert!(home.workspace_dir("default").join("stm.sqlite").exists());
}

/// `.env` in the home never overrides variables already set in the process.
#[test]
fn process_env_wins_over_home_dotenv() {
    let llm = FakeLlm::start();
    let home = Home::new(&llm.base_url());
    std::fs::write(home.path().join(".env"), "OPENAI_API_KEY=home-key\n").unwrap();
    let home_str = home.path().display().to_string();
    let mut server = spawn_raw(
        &home,
        &["mcp"],
        &[
            ("NEUROLITHE_HOME", home_str.as_str()),
            ("OPENAI_API_KEY", API_KEY),
        ],
    );
    server.initialize();
    assert!(
        !server
            .call_tool("store_memory", json!({"fact_text": "x"}))
            .is_error
    );
    assert!(
        auths(&llm)
            .iter()
            .all(|a| a == &format!("Bearer {API_KEY}")),
        "{:?}",
        auths(&llm)
    );
}

#[test]
fn home_flag_wins_over_env_and_works_on_either_side_of_the_subcommand() {
    let llm = FakeLlm::start();
    let env_home = Home::new(&llm.base_url());
    let flag_home = Home::new(&llm.base_url());
    let flag = flag_home.path().display().to_string();

    for args in [
        vec!["--home", flag.as_str(), "mcp"],
        vec!["mcp", "--home", flag.as_str()],
    ] {
        // env_home.spawn_mcp sets NEUROLITHE_HOME=env_home; the flag must win.
        let mut server = McpProcess::spawn_with(
            Path::new(bin()),
            &args,
            env_home.cwd.path(),
            env_home.cwd.path(),
            &env_home
                .env()
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect::<Vec<_>>(),
        );
        server.initialize();
        let cur = server.call_tool("workspace_current", json!({})).json();
        let path = std::fs::canonicalize(cur["path"].as_str().unwrap()).unwrap();
        assert!(
            path.starts_with(std::fs::canonicalize(flag_home.path()).unwrap()),
            "{args:?}: {path:?}"
        );
    }
    assert!(
        !env_home.path().join("workspaces").exists(),
        "NEUROLITHE_HOME used despite --home"
    );
}

/// (CLI args, extra env) for one config-selection case.
type Case<'a> = (Vec<&'a str>, Vec<(&'a str, &'a str)>);

#[test]
fn config_flag_and_env_select_the_config_file() {
    let llm = FakeLlm::start();
    let home = Home::empty();
    let alt = home.cwd.path().join("alt.toml");
    std::fs::write(
        &alt,
        format!(
            "workspace = \"alt-ws\"\n{}",
            home_config_toml(&llm.base_url())
        ),
    )
    .unwrap();
    let alt_str = alt.display().to_string();

    let cases: [Case; 2] = [
        (vec!["--config", &alt_str, "mcp"], vec![]),
        (vec!["mcp"], vec![("NEUROLITHE_CONFIG", &alt_str)]),
    ];
    for (args, extra) in cases {
        let mut env: Vec<(String, String)> = home.env();
        env.extend(extra.iter().map(|(k, v)| (k.to_string(), v.to_string())));
        let env_ref: Vec<(&str, &str)> =
            env.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        let mut server = spawn_raw(&home, &args, &env_ref);
        server.initialize();
        let cur = server.call_tool("workspace_current", json!({}));
        assert_eq!(
            cur.json()["name"],
            "alt-ws",
            "{args:?} {extra:?}: {}",
            cur.text
        );
    }
}

#[test]
fn explicit_missing_config_is_an_error_but_missing_default_is_not() {
    let home = Home::empty();
    let missing = home.cwd.path().join("nope.toml");
    let out = home.cli(&["--config", missing.to_str().unwrap(), "workspace", "list"]);
    assert_eq!(out.code, Some(1), "{out:?}");
    assert!(
        out.stderr.contains("Error: config file not found"),
        "{}",
        out.stderr
    );

    // No <home>/neurolithe.toml at all: defaults apply, the CLI works.
    let out = home.cli(&["workspace", "list"]);
    out.assert_success();
}

#[test]
fn standalone_without_subcommand_prints_help_and_exits_2() {
    if cfg!(feature = "kafka") {
        return; // a kafka build runs the daemon instead
    }
    let home = Home::empty();
    let out = home.cli(&[]);
    assert_eq!(out.code, Some(2), "{out:?}");
    let all = format!("{}{}", out.stdout, out.stderr).to_lowercase();
    assert!(all.contains("mcp") && all.contains("usage"), "{all}");
}

#[test]
fn stale_store_path_config_is_ignored_with_a_warning() {
    let llm = FakeLlm::start();
    let home = Home::empty();
    let stale = home.cwd.path().join("stale-stm.sqlite");
    std::fs::write(
        home.path().join("neurolithe.toml"),
        home_config_toml(&llm.base_url()) + &format!("\n[stm]\npath = \"{}\"\n", stale.display()),
    )
    .unwrap();
    let mut server = home.spawn_mcp(&[], &[]);
    server.initialize();
    assert!(
        !server
            .call_tool("store_memory", json!({"fact_text": "x"}))
            .is_error
    );
    assert!(!stale.exists(), "stale [stm].path was used");
    assert!(home.workspace_dir("default").join("stm.sqlite").exists());
    let (_, stderr) = server.shutdown();
    assert!(
        stderr.to_lowercase().contains("warn"),
        "no warning: {stderr}"
    );
}

#[cfg(unix)]
#[test]
fn home_and_store_files_are_private() {
    use std::os::unix::fs::PermissionsExt;
    let llm = FakeLlm::start();
    let home = Home::empty();
    let root = home.path().join("fresh-home");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::remove_dir(&root).unwrap(); // must be created by the binary
    let root_str = root.display().to_string();
    let cfg = home.cwd.path().join("cfg.toml");
    std::fs::write(&cfg, home_config_toml(&llm.base_url())).unwrap();
    let cfg_str = cfg.display().to_string();

    let mut server = spawn_raw(
        &home,
        &["--home", &root_str, "--config", &cfg_str, "mcp"],
        &[("OPENAI_API_KEY", API_KEY)],
    );
    server.initialize();
    assert!(
        !server
            .call_tool("store_memory", json!({"fact_text": "x"}))
            .is_error
    );

    let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(&root), 0o700, "home dir mode");
    let ws = root.join("workspaces").join("default");
    assert_eq!(mode(&ws), 0o700, "workspace dir mode");
    for f in ["stm.sqlite", "ltm.sqlite"] {
        assert_eq!(mode(&ws.join(f)), 0o600, "{f} mode");
    }
}

// --- init --------------------------------------------------------------------

/// Env that points the embedder at the fake LLM. `init` pre-downloads the
/// default local model (~130 MB) when the resolved embedder is `local`; default
/// test runs must never download, so every init test resolves to the fake.
fn fake_embedder_env(llm: &FakeLlm) -> Vec<(String, String)> {
    let url = llm.base_url();
    vec![
        (
            "NEUROLITHE__LLM__EMBEDDING_PROVIDER".into(),
            "openai".into(),
        ),
        (
            "NEUROLITHE__LLM__EMBEDDING_MODEL".into(),
            "fake-embed".into(),
        ),
        ("NEUROLITHE__LLM__EMBEDDING_BASE_URL".into(), url),
        ("OPENAI_API_KEY".into(), API_KEY.into()),
    ]
}

/// The JSON block `init` prints (from the first `{` to the end of stdout).
fn init_snippet(stdout: &str) -> Value {
    let start = stdout
        .find('{')
        .unwrap_or_else(|| panic!("no JSON in:\n{stdout}"));
    serde_json::from_str(stdout[start..].trim())
        .unwrap_or_else(|e| panic!("bad JSON snippet ({e}):\n{stdout}"))
}

#[test]
fn init_writes_config_creates_workspace_and_prints_mcp_snippet() {
    let llm = FakeLlm::start();
    let home = Home::empty();
    let root = home.path().join("h");
    let root_str = root.display().to_string();
    let out = run_cli(
        &["--home", &root_str, "--workspace", "proj", "init"],
        home.cwd.path(),
        &fake_embedder_env(&llm),
    );
    out.assert_success();
    // A non-local embedder means no model download.
    assert!(!root.join("models").exists(), "init downloaded a model");
    for label in ["Config:", "Home:", "Workspace:"] {
        assert!(
            out.stdout.contains(label),
            "missing {label}:\n{}",
            out.stdout
        );
    }

    let cfg = root.join("neurolithe.toml");
    let text = std::fs::read_to_string(&cfg).expect("config written");
    let example = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/neurolithe.example.toml"
    ))
    .unwrap();
    assert_eq!(text, example, "init must write the example config verbatim");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&cfg).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "config mode");
    }
    assert!(
        root.join("workspaces").join("proj").is_dir(),
        "workspace not created"
    );

    let snippet = init_snippet(&out.stdout);
    let entry = &snippet["mcpServers"]["neurolithe"];
    let command = entry["command"].as_str().expect("command");
    assert!(
        Path::new(command).is_absolute(),
        "command not absolute: {command}"
    );
    assert_eq!(
        std::fs::canonicalize(command).unwrap(),
        std::fs::canonicalize(bin()).unwrap()
    );
    let args: Vec<String> = serde_json::from_value(entry["args"].clone()).unwrap();
    assert_eq!(args[0], "mcp", "{args:?}");
    let pos = args
        .iter()
        .position(|a| a == "--home")
        .expect("--home in args");
    assert!(Path::new(&args[pos + 1]).is_absolute(), "{args:?}");
    assert_eq!(
        std::fs::canonicalize(&args[pos + 1]).unwrap(),
        std::fs::canonicalize(&root).unwrap()
    );
    let pos = args
        .iter()
        .position(|a| a == "--workspace")
        .expect("--workspace in args");
    assert_eq!(args[pos + 1], "proj");
    // The CWD is untouched.
    assert_eq!(std::fs::read_dir(home.cwd.path()).unwrap().count(), 0);
}

#[test]
fn init_keeps_an_existing_config() {
    let home = Home::empty();
    let cfg = home.path().join("neurolithe.toml");
    std::fs::write(&cfg, "# my own config\n").unwrap();
    let llm = FakeLlm::start();
    let extra = fake_embedder_env(&llm);
    let extra: Vec<(&str, &str)> = extra
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let out = home.cli_env(&["init"], &extra);
    out.assert_success();
    assert!(out.stdout.contains("kept"), "{}", out.stdout);
    assert_eq!(std::fs::read_to_string(&cfg).unwrap(), "# my own config\n");
    // Without --workspace / non-default config, args carry only the home.
    let snippet = init_snippet(&out.stdout);
    let args: Vec<String> =
        serde_json::from_value(snippet["mcpServers"]["neurolithe"]["args"].clone()).unwrap();
    assert!(!args.contains(&"--workspace".to_string()), "{args:?}");
    assert!(!args.contains(&"--config".to_string()), "{args:?}");
}
