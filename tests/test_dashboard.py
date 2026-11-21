"""Streamlit script tests execute each page and independent laboratory controls."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

@pytest.mark.parametrize("page", ["Overview","Audio Classification","Signal Analysis","DSP Laboratory","Classification Evaluation"])
def test_dashboard_page(page):
    app=AppTest.from_file(str(Path(__file__).parents[1]/"app.py"),default_timeout=30).run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception
    assert not app.error

def test_lab_filter_controls():
    app=AppTest.from_file(str(Path(__file__).parents[1]/"app.py"),default_timeout=30).run()
    app.sidebar.radio[0].set_value("DSP Laboratory").run()
    app.checkbox[0].check().run()
    assert not app.exception and not app.error
    [s for s in app.selectbox if s.label=="Sampling frequency (Hz)"][0].set_value(44100).run()
    [s for s in app.selectbox if s.label=="Window length (samples)"][0].set_value(256).run()
    assert not app.exception and not app.error
    [s for s in app.selectbox if s.label=="Filter type"][0].set_value("bandpass").run()
    assert not app.exception and not app.error
