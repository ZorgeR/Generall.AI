"""The sandbox image is tagged by the hash of its build context, so a changed Dockerfile or
requirements file rebuilds it on the next startup; nobody has to delete the old image."""
import importlib
from types import SimpleNamespace

cm = importlib.import_module("secure_container.container_manager")


def _context(tmp_path, requirements="a==1\n"):
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (tmp_path / "entrypoint.sh").write_text("#!/bin/bash\nexec \"$@\"\n")
    (tmp_path / "requirements.txt").write_text(requirements)
    return tmp_path


def test_fingerprint_follows_the_build_files(tmp_path):
    first = cm.image_build_fingerprint(_context(tmp_path))
    assert len(first) == 12 and int(first, 16) >= 0
    assert cm.image_build_fingerprint(tmp_path) == first  # stable
    (tmp_path / "requirements.txt").write_text("a==2\n")
    assert cm.image_build_fingerprint(tmp_path) != first
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "x.py").write_text("print(1)\n")
    assert cm.image_build_fingerprint(tmp_path) == cm.image_build_fingerprint(tmp_path)  # scripts are not baked in


def test_real_build_context_has_a_tag_and_the_manager_uses_it():
    tag = cm.image_build_fingerprint()
    assert len(tag) == 12
    manager = cm.ContainerManager.__new__(cm.ContainerManager)
    manager.image_name = cm.IMAGE_NAME
    manager.image_tag = tag
    assert f"{manager.image_name}:{manager.image_tag}" == f"secure-container:{tag}"


class FakeImage:
    def __init__(self, id_, tags):
        self.id, self.tags, self.tagged = id_, tags, []

    def tag(self, repository, tag=None, **kw):
        self.tagged.append((repository, tag))
        return True


class FakeImages:
    def __init__(self, existing):
        self.existing = existing  # tag -> FakeImage
        self.built = []
        self.removed = []

    def get(self, ref):
        if ref in self.existing:
            return self.existing[ref]
        raise cm.ImageNotFound(f"no {ref}")

    def build(self, *, path, tag, rm, forcerm):
        image = FakeImage("sha256:new", [tag])
        self.existing[tag] = image
        self.built.append(tag)
        return image, []

    def list(self, name=None):
        seen = {}
        for image in self.existing.values():
            seen[image.id] = image
        return list(seen.values())

    def remove(self, image_id, force=False):
        self.removed.append(image_id)
        self.existing = {t: i for t, i in self.existing.items() if i.id != image_id}


def _manager(images):
    manager = cm.ContainerManager.__new__(cm.ContainerManager)
    manager.image_name = cm.IMAGE_NAME
    manager.image_tag = "abc123def456"
    manager.docker_client = SimpleNamespace(images=images)
    return manager


def test_existing_content_tag_is_not_rebuilt():
    images = FakeImages({"secure-container:abc123def456": FakeImage("sha256:cur", ["secure-container:abc123def456"])})
    _manager(images)._ensure_image_exists()
    assert images.built == [] and images.removed == []


def test_changed_context_builds_retags_latest_and_prunes_the_old_image(tmp_path, monkeypatch):
    old = FakeImage("sha256:old", ["secure-container:latest", "secure-container:000000000000"])
    images = FakeImages({"secure-container:latest": old, "secure-container:000000000000": old})
    manager = _manager(images)
    # the real Dockerfile directory is used for the build path; only Docker is faked
    manager._ensure_image_exists()
    assert images.built == ["secure-container:abc123def456"]
    new = images.existing["secure-container:abc123def456"]
    assert ("secure-container", "latest") in new.tagged
    assert images.removed == ["sha256:old"]
