from flask import Flask, request, jsonify
import cv2
import torch
from gfpgan import GFPGANer
import numpy as np
import os
from basicsr.utils import imwrite
from PIL import Image
import base64

app = Flask(__name__)


@app.route("/restore", methods=["POST"])
def process_image():
    raw_files = request.files.to_dict(flat=False)
    files=raw_files['files']
    img_list=[]
    for filei in files:
        file_bytes = np.fromfile(filei, np.uint8)
        #   # convert numpy array to image
        cv2_img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        img_list.append(cv2_img)

    payload = request.form.to_dict()
    version='1.3'
    upscale=2
    bg_upsampler='realesrgan'
    bg_title=400
    center_face=False
    aligned=False
    ext='auto'
    weight=0.5

    if 'version' in payload.keys():
        version=payload['version']
    if 'upscale' in payload.keys():
        upscale=int(payload['upscale'])
    if 'bg_upsampler' in payload.keys():
        bg_upsampler=payload['bg_upsampler']
    if 'bg_title' in payload.keys():
        bg_title=int(payload['bg_title'])
    if 'center_face' in payload.keys():
        center_face=bool(payload['center_face'])
    if 'aligned' in payload.keys():
        aligned=bool(payload['aligned'])
    if 'ext' in payload.keys():
        ext=payload['ext']
    if 'weight' in payload.keys():
        weight=int(payload['weight'])
    # Read the image via file.stream

    if version == '1':
        arch = 'original'
        channel_multiplier = 1
        model_name = 'GFPGANv1'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
    elif version == '1.2':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANCleanv1-NoCE-C2'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
    elif version == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    elif version == '1.4':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    elif version == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    else:
        return jsonify({msg:'ERROR WRONG VERSION NUMBER'})

    # API CORE BASED ON GFPGANS #

    if bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=args.bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
    else:
        bg_upsampler = None



    model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)
    output = []
    for img in img_list:
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            img,
            has_aligned=aligned,
            only_center_face=center_face,
            paste_back=True,
            weight=weight)
        cropped_faces_f=[base64.b64encode(face) for face in cropped_faces]
        restored_faces_f=[base64.b64encode(face) for face in restored_faces]
        restored_img_f=base64.b64encode(restored_img)
        raw_out={'cropped_faces':str(cropped_faces_f),'restored_faces':str(restored_faces_f),'restored_img':str(restored_img_f)}
        output.append(raw_out)

    return jsonify({'output':output})


if __name__ == "__main__":
    app.run(debug=True)
