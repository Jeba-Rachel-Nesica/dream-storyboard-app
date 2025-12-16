from app.pipelines.beats import extract_beats

def test_extract_beats():
    script = "A calm dream. The dreamer walks in a garden. The sun is shining. Birds sing. The dreamer feels safe."
    beats = extract_beats(script)
    assert 4 <= len(beats) <= 8
    assert 'safe' in beats[-1]['scene_text'].lower() or 'peace' in beats[-1]['scene_text'].lower() or 'comfort' in beats[-1]['scene_text'].lower()
