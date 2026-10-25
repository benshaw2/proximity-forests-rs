import platform

import proximity_forests


def _pep440_from_cargo(version):
    return version.replace("-alpha.", "a").replace("-beta.", "b").replace("-rc.", "rc")


def _normalized_architecture(value):
    aliases = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "aarch64",
        "aarch64": "aarch64",
    }
    normalized = value.strip().lower()
    return aliases.get(normalized, normalized)


def test_native_module_imports():
    native_version, native_architecture = proximity_forests.runtime_info()
    assert _pep440_from_cargo(native_version) == proximity_forests.__version__
    assert _normalized_architecture(native_architecture) == _normalized_architecture(
        platform.machine()
    )
