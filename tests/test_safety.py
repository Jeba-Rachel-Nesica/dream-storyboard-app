from app.pipelines.safety import is_safe_text

def test_safety_filter_blocks_unsafe():
    unsafe = "This scene contains violence and blood."
    assert not is_safe_text(unsafe)
    safe = "A peaceful garden."
    assert is_safe_text(safe)
