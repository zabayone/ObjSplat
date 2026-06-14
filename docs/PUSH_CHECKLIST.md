# Push Checklist

This workspace root is not currently a Git repository. Before pushing, decide how to handle the nested repositories:

- `GUI/` is a Git checkout of `https://github.com/playcanvas/supersplat.git` with local ObjSplat UI changes.
- `submodules/splat-apple/` is a Git checkout of `https://github.com/ghif/splat-apple.git` with local MLX changes.
- `submodules/deva/.git` points to a missing parent `.git/modules/...` path and should not be committed as-is.

## Recommended Thesis Repo Layout

For the thesis project, the simplest pushable layout is to vendor the modified code in the main ObjSplat repo:

```bash
# From the ObjSplat root
mv GUI/.git GUI/.git.upstream-backup
mv submodules/splat-apple/.git submodules/splat-apple/.git.upstream-backup
rm -f submodules/deva/.git

git init
git add .
git status --short
git commit -m "Initial ObjSplat thesis pipeline"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

This includes the locally modified GUI and Splat-Apple code as normal source files.

## Alternative: True Submodules

Use this only if you plan to push your GUI/Splat-Apple changes to forks first:

```bash
git init
git submodule add <your-supersplat-fork-url> GUI
git submodule add <your-splat-apple-fork-url> submodules/splat-apple
git add .
git commit -m "Initial ObjSplat thesis pipeline"
```

In this mode, the main repo stores only submodule commit pointers. Unpushed local changes inside those folders will not be included.

## Before Committing

Check that generated assets and model weights are ignored:

```bash
git status --short --ignored
```

The following should remain ignored:

- `checkpoints/*` except `checkpoints/README.md`
- `outputs_lgs/`
- `GUI/node_modules/`
- `GUI/dist/`
- `__pycache__/`
- generated `.ply`, `.npy`, `.npz`, `.pth`, `.pt`, `.safetensors`

Run quick checks:

```bash
python -m py_compile gen_layerdata_from_deva.py run_layered_deva_pipeline.py mps_splat_backend.py LayerPano.py
cd GUI && npm run build
```
