from importlib.metadata import version

import proximity_forests


def test_distribution_version_is_public_version():
    assert proximity_forests.__version__ == version("proximity-forests")
    #assert proximity_forests.__version__ == "0.1.0a1"
