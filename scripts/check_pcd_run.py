import sys
import glob
import traceback

import numpy as np
from plyfile import PlyData


def check_pcd(path):
    ply = PlyData.read(path)
    v = ply['vertex']
    xyz = np.stack([v['x'], v['y'], v['z']], axis=1)
    # RGB may be named differently; try common variants
    color_cols = None
    for cset in (('red','green','blue'), ('r','g','b'), ('nx','ny','nz')):
        try:
            rgb = np.stack([v[cset[0]], v[cset[1]], v[cset[2]]], axis=1)
            color_cols = rgb
            break
        except Exception:
            pass

    print(f"=== {path} ===")
    print(f"Num punti:     {len(xyz):,}")
    print(f"Bbox XYZ:      {np.round(xyz.min(axis=0),3)} -> {np.round(xyz.max(axis=0),3)}")
    print(f"Range per asse: X={np.ptp(xyz[:,0]):.3f}  Y={np.ptp(xyz[:,1]):.3f}  Z={np.ptp(xyz[:,2]):.3f}")
    print(f"NaN/Inf:        {(~np.isfinite(xyz)).sum()} valori non finiti")
    if color_cols is not None:
        rgb = color_cols
        print(f"Colori range:  {rgb.min()} -> {rgb.max()} (dovrebbe essere 0-255)" )
    else:
        print("Colori: non trovati nelle proprietà standard")

    try:
        from scipy.spatial import cKDTree
        n_check = min(50000, len(xyz))
        tree = cKDTree(xyz[:n_check])
        d, _ = tree.query(xyz[:n_check], k=2)
        nn = d[:, 1]
        print(f"Distanza media  NN: {nn.mean():.5f}")
        print(f"Distanza mediana NN: {np.median(nn):.5f}")
        print(f"Distanza max NN:   {nn.max():.5f}")
        print(f"Punti con NN > 0.5: {(nn > 0.5).sum()} (outlier lontani)")
        print(f"Punti duplicati (NN < 1e-5): {(nn < 1e-5).sum()}")
    except Exception as e:
        print("Attenzione: scipy.spatial.cKDTree non disponibile o errore nel calcolo dei vicini:", e)


if __name__ == '__main__':
    paths = sys.argv[1:]
    if not paths:
        patterns = [
            'outputs_lgs/traindata/**/pcd_rgb_layer*.ply',
            'outputs_lgs/traindata/**/pcd_mask_layer*.ply',
            'outputs_lgs/layering/pcd_rgb.ply',
            'outputs_lgs/scene/gsplat_layer*.ply',
            'results/**/ply/*.ply',
            'src/**/demo*.ply'
        ]
        for p in patterns:
            found = glob.glob(p, recursive=True)
            for f in found:
                paths.append(f)

    if not paths:
        print('Nessun file PLY trovato con i pattern standard. Fornisci percorsi come argomenti.')
        sys.exit(1)

    for p in sorted(set(paths)):
        try:
            check_pcd(p)
            print('\n')
        except Exception:
            print(f'Errore durante l\'analisi di {p}:')
            traceback.print_exc()
            print('\n')
