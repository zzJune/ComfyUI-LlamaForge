"""Local llama-server builds and GGUF model discovery."""

import json
import os
import re
import shutil
from pathlib import Path

import folder_paths


ROOT = Path(__file__).resolve().parent
ENGINE_FILENAME = "llama-server.exe" if os.name == "nt" else "llama-server"
MODEL_CATEGORY = "llamaforge"
NO_ENGINE = "(configure llama-server)"
NO_MODEL = "(add a GGUF model)"
NO_PROJECTOR = "(auto / embedded)"
SHARD_SUFFIX = re.compile(r"-(\d{5})-of-\d{5}\.gguf$", re.IGNORECASE)


def register_model_directories():
    roots = [Path(folder_paths.models_dir) / "LLM", Path(folder_paths.models_dir) / "llamaforge"]
    if "LLM" in folder_paths.folder_names_and_paths:
        roots.extend(Path(p) for p in folder_paths.get_folder_paths("LLM"))
    extra = os.environ.get("LLAMAFORGE_MODEL_DIRS", "")
    roots.extend(Path(p.strip()) for p in extra.split(os.pathsep) if p.strip())
    for root in dict.fromkeys(root.expanduser().resolve() for root in roots):
        folder_paths.add_model_folder_path(MODEL_CATEGORY, str(root))


def available_models():
    names = folder_paths.get_filename_list(MODEL_CATEGORY)
    models = []
    for name in names:
        filename = Path(name).name
        if not filename.lower().endswith(".gguf") or "mmproj" in filename.lower():
            continue
        match = SHARD_SUFFIX.search(filename)
        if match and int(match.group(1)) != 1:
            continue
        models.append(name)
    return sorted(models)


def available_projectors():
    """Discover separately packaged multimodal projector GGUFs."""
    return sorted(
        name for name in folder_paths.get_filename_list(MODEL_CATEGORY)
        if Path(name).name.lower().endswith(".gguf")
        and "mmproj" in Path(name).name.lower()
    )


def resolve_projector(name):
    if name not in available_projectors():
        raise ValueError(f"LlamaForge: unknown mmproj {name!r}; refresh the node's projector list.")
    path = folder_paths.get_full_path(MODEL_CATEGORY, name)
    if not path:
        raise FileNotFoundError(f"LlamaForge: projector file is missing: {name}")
    resolved = Path(path).resolve(strict=True)
    if not any(resolved.is_relative_to(Path(root).resolve())
               for root in folder_paths.get_folder_paths(MODEL_CATEGORY)):
        raise ValueError("LlamaForge: projector path escapes the configured model directories.")
    return resolved


def resolve_model(name):
    if name not in available_models():
        raise ValueError(f"LlamaForge: unknown GGUF model {name!r}; refresh the node's model list.")
    path = folder_paths.get_full_path(MODEL_CATEGORY, name)
    if not path:
        raise FileNotFoundError(f"LlamaForge: model is missing: {name}")
    resolved = Path(path).resolve(strict=True)
    if not any(resolved.is_relative_to(Path(root).resolve()) for root in folder_paths.get_folder_paths(MODEL_CATEGORY)):
        raise ValueError("LlamaForge: model path escapes the configured model directories.")
    return resolved


def _executables_in(folder):
    if not folder.is_dir():
        return
    candidates = [
        folder / ENGINE_FILENAME,
        folder / "bin" / ENGINE_FILENAME,
        *folder.glob("*/" + ENGINE_FILENAME),
        *folder.glob("*/bin/" + ENGINE_FILENAME),
    ]
    for path in candidates:
        if path.is_file():
            yield path


def available_engines():
    """Return a stable, explicit name -> absolute executable path registry."""
    engines = {}
    settings = ROOT / "engines.json"
    if settings.is_file():
        data = json.loads(settings.read_text(encoding="utf-8"))
        versions = data.get("versions", {})
        if not isinstance(versions, dict):
            raise ValueError("LlamaForge: engines.json must contain a 'versions' object.")
        for name, raw_path in versions.items():
            if not isinstance(name, str) or not isinstance(raw_path, str) or not name.strip():
                raise ValueError("LlamaForge: engine names and paths must be strings.")
            path = Path(os.path.expandvars(raw_path)).expanduser()
            if not path.is_absolute():
                path = ROOT / path
            engines[name] = path.resolve()

    engine_dirs = [ROOT / "engines"]
    extra = os.environ.get("LLAMAFORGE_ENGINE_DIRS", "")
    engine_dirs.extend(Path(part.strip()).expanduser() for part in extra.split(os.pathsep) if part.strip())
    for directory in engine_dirs:
        for path in _executables_in(directory):
            key = path.parent.name if path.parent.name != "bin" else path.parent.parent.name
            engines.setdefault(key, path.resolve())

    explicit = os.environ.get("LLAMAFORGE_LLAMA_SERVER")
    if explicit:
        engines.setdefault("environment", Path(explicit).expanduser().resolve())
    on_path = shutil.which(ENGINE_FILENAME)
    if on_path:
        engines.setdefault("PATH", Path(on_path).resolve())
    return {name: path for name, path in sorted(engines.items()) if path.is_file()}


def resolve_engine(name):
    engine = available_engines().get(name)
    if engine is None:
        raise FileNotFoundError(
            f"LlamaForge: llama.cpp build {name!r} is unavailable. "
            "Place llama-server in engines/<version>/ or configure engines.json."
        )
    return engine


register_model_directories()
