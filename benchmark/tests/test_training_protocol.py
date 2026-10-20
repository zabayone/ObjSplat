import json, tempfile, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from mps_splat_backend import extract_gaussian_params_from_ply
import unittest


class TrainingProtocolSmokeTests(unittest.TestCase):
    def test_sampling_parity_and_seeded_metal_baseline(self):
        try:
            import mlx.core
        except ImportError as exc:
            raise unittest.SkipTest(str(exc))
        from LayerPano import LayerPano
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            layer=root/'traindata/layer0'; (layer/'frames').mkdir(parents=True)
            frames=root/'traindata/perspective_frames/frames'; frames.mkdir(parents=True)
            (root/'scene').mkdir()
            Image.new('RGB',(64,32)).save(root/'rgb.png')
            metadata={'layer_groups':[{'layer_idx':0,'instance_ids':[1]}], 'benchmark_view_split':{'training_indices':[0],'evaluation_indices':[1]}}
            (root/'traindata/layer_instances.json').write_text(json.dumps(metadata))
            data=np.zeros(40,dtype=[(k,'f4') for k in ['x','y','z']]+[(k,'u1') for k in ['red','green','blue']]+[('label','i4')])
            data['x']=np.linspace(-0.3,0.3,40); data['z']=-2; data['red']=100; data['green']=120; data['blue']=140;data['label']=1
            for kind in ('rgb','mask'):
                PlyData([PlyElement.describe(data,'vertex')]).write(layer/f'pcd_{kind}_layer0.ply')
            Image.new('L',(64,32),255).save(layer/'layer0_erp_mask.png')
            for index in (0,1):
                Image.new('RGB',(32,32),(100,120,140)).save(frames/f'rgb_{index}.png')
                np.save(frames/f'transform_matrix_{index}.npy',np.eye(4))
            Image.new('RGB',(32,32),(100,120,140)).save(layer/'frames/rgb_0.png')
            Image.new('L',(32,32),255).save(layer/'frames/mask_0.png')
            np.save(layer/'frames/transform_matrix_0.npy',np.eye(4))
            driver=LayerPano(str(root/'scene'),backend='splat-apple',downsample_ratio=0.5,max_points=0,no_adaptive=True,layer_iterations=2,training_image_size=32,min_gaussian_retention=.95)
            output=driver.create_layer_instances(str(root),10)
            layered,_=extract_gaussian_params_from_ply(output[0]);assert len(layered['means'])==20
            for run in (1,2):
                subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'monolithic_baseline.py'),'--scene_root',str(root),'--output',str(root/f'mono{run}.ply'),'--iterations','2','--image_size','32','--downsample_ratio','.5','--seed','42'],check=True, capture_output=True, text=True)
            one,_=extract_gaussian_params_from_ply(str(root/'mono1.ply'));two,_=extract_gaussian_params_from_ply(str(root/'mono2.ply'))
            assert len(one['means'])==20
            for key in one: np.testing.assert_array_equal(one[key],two[key])
            print('PASS: layered and monolithic retain 20/40 points; two seeded Metal baseline runs are identical')
