//! Tool-argument parsing. Every problem becomes a clear `isError` tool result
//! naming the argument; nothing is silently coerced.

use crate::domain::models::TimeFilter;
use serde_json::Value;

use super::limits::MAX_ID_BYTES;

/// A caller mistake, reported back as an `isError` tool result.
pub type ArgResult<T> = Result<T, String>;

/// Tool arguments. A missing `arguments` is treated as `{}`.
pub struct Args<'a>(pub &'a serde_json::Map<String, Value>);

impl<'a> Args<'a> {
    pub fn get(&self, name: &str) -> Option<&'a Value> {
        self.0.get(name).filter(|v| !v.is_null())
    }

    /// Names of the arguments present (non-null).
    pub fn present(&self) -> impl Iterator<Item = &'a str> + 'a {
        let map = self.0;
        map.iter()
            .filter(|(_, v)| !v.is_null())
            .map(|(k, _)| k.as_str())
    }

    /// An optional string, type-checked and size-capped. Blank → `None`.
    pub fn opt_str(&self, name: &str, max_bytes: usize) -> ArgResult<Option<&'a str>> {
        match self.get(name) {
            None => Ok(None),
            Some(Value::String(s)) => {
                if s.len() > max_bytes {
                    Err(format!(
                        "'{name}' is too large ({} bytes; max {max_bytes})",
                        s.len()
                    ))
                } else if s.trim().is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(s.as_str()))
                }
            }
            Some(_) => Err(format!("'{name}' must be a string")),
        }
    }

    /// A required, non-blank, size-capped string.
    pub fn req_str(&self, name: &str, max_bytes: usize) -> ArgResult<&'a str> {
        self.opt_str(name, max_bytes)?
            .ok_or_else(|| format!("'{name}' is required and must be a non-empty string"))
    }

    /// An optional non-negative integer clamped to `[min, max]`. Negative or
    /// fractional values are rejected — never cast (a negative `k` used to
    /// wrap to "no limit", SEC-12).
    pub fn count(&self, name: &str, default: usize, min: usize, max: usize) -> ArgResult<usize> {
        match self.get(name) {
            None => Ok(default),
            Some(v) => match v.as_u64() {
                Some(n) => Ok(usize::try_from(n).unwrap_or(max).clamp(min, max)),
                None => Err(format!("'{name}' must be a non-negative integer")),
            },
        }
    }

    /// An optional count with no default (`None` when absent).
    pub fn opt_count(&self, name: &str, min: usize, max: usize) -> ArgResult<Option<usize>> {
        match self.get(name) {
            None => Ok(None),
            Some(_) => self.count(name, 0, min, max).map(Some),
        }
    }

    /// A required integer id (may be any i64).
    pub fn req_id(&self, name: &str) -> ArgResult<i64> {
        self.get(name)
            .and_then(|v| v.as_i64())
            .ok_or_else(|| format!("'{name}' is required and must be an integer"))
    }

    /// An optional array of strings (each size-capped).
    pub fn str_list(&self, name: &str, max_items: usize) -> ArgResult<Vec<String>> {
        match self.get(name) {
            None => Ok(Vec::new()),
            Some(Value::Array(items)) => {
                if items.len() > max_items {
                    return Err(format!("'{name}' has too many items (max {max_items})"));
                }
                items
                    .iter()
                    .map(|v| match v.as_str() {
                        Some(s) if s.len() <= MAX_ID_BYTES => Ok(s.to_string()),
                        Some(_) => Err(format!("an item of '{name}' is too long")),
                        None => Err(format!("'{name}' must be an array of strings")),
                    })
                    .collect()
            }
            Some(_) => Err(format!("'{name}' must be an array of strings")),
        }
    }

    /// `time_filter: {after?, before?}` with `YYYY-MM-DD[...]` dates. Malformed
    /// input is an error, not silently ignored (QA-9).
    pub fn time_filter(&self) -> ArgResult<TimeFilter> {
        let Some(v) = self.get("time_filter") else {
            return Ok(TimeFilter::default());
        };
        let Some(obj) = v.as_object() else {
            return Err("'time_filter' must be an object {after?, before?}".into());
        };
        let date = |key: &str| -> ArgResult<Option<String>> {
            match obj.get(key).filter(|v| !v.is_null()) {
                None => Ok(None),
                Some(Value::String(s)) if is_iso_date_prefix(s) => Ok(Some(s.clone())),
                Some(_) => Err(format!(
                    "'time_filter.{key}' must be a date string like 2026-01-31"
                )),
            }
        };
        Ok(TimeFilter {
            after: date("after")?,
            before: date("before")?,
        })
    }
}

/// `YYYY-MM-DD`, optionally followed by a time part (≤ 40 chars total).
fn is_iso_date_prefix(s: &str) -> bool {
    let b = s.as_bytes();
    s.len() <= 40
        && b.len() >= 10
        && b[..4].iter().all(u8::is_ascii_digit)
        && b[4] == b'-'
        && b[5..7].iter().all(u8::is_ascii_digit)
        && b[7] == b'-'
        && b[8..10].iter().all(u8::is_ascii_digit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// SEC-12: huge counts clamp to the max, zero clamps to the min, absent
    /// takes the default; negative and fractional values are errors.
    #[test]
    fn test_count_clamps_and_rejects() {
        let empty = serde_json::Map::new();
        let mut m = serde_json::Map::new();
        m.insert("k".into(), json!(u64::MAX));
        assert_eq!(Args(&m).count("k", 10, 1, 100), Ok(100));
        m.insert("k".into(), json!(0));
        assert_eq!(Args(&m).count("k", 10, 1, 100), Ok(1));
        assert_eq!(Args(&empty).count("k", 10, 1, 100), Ok(10));
        for bad in [json!(-5), json!(2.5), json!("3")] {
            m.insert("k".into(), bad.clone());
            let err = Args(&m).count("k", 10, 1, 100).unwrap_err();
            assert!(err.contains("'k'"), "{bad}: {err}");
        }
    }

    #[test]
    fn test_iso_dates() {
        assert!(is_iso_date_prefix("2026-01-31"));
        assert!(is_iso_date_prefix("2026-01-31T10:00:00Z"));
        assert!(!is_iso_date_prefix("yesterday"));
        assert!(!is_iso_date_prefix("2026-1-31"));
    }
}
