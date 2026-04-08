use hyverk_sandbox::{verify_response, VerificationStage};

#[tokio::main]
async fn main() {
    let good = r#"```rust
/// Check if n is prime
pub fn is_prime(n: u64) -> bool {
    if n <= 1 { return false; }
    for i in 2..=((n as f64).sqrt() as u64) {
        if n % i == 0 { return false; }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_prime() {
        assert!(is_prime(7));
        assert!(!is_prime(4));
        assert!(!is_prime(1));
    }
}
```"#;

    let result = hyverk_sandbox::verify_response(good, None).await;
    println!("Good code:  passed={}, stage={:?}, tests_in_code={}, ms={}",
        result.passed, result.stage, result.signals.has_tests, result.duration_ms);

    let bad = "```rust\nfn broken( {\n    let x = ;\n}\n```";
    let result2 = hyverk_sandbox::verify_response(bad, None).await;
    println!("Bad code:   passed={}, stage={:?}", result2.passed, result2.stage);

    let no_code = "Here is some text without code blocks.";
    let result3 = hyverk_sandbox::verify_response(no_code, None).await;
    println!("No code:    passed={}, stage={:?}", result3.passed, result3.stage);
}
