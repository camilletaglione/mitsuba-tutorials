import numpy as np
import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
import os
import cv2
#import xml.etree.ElementTree as ET


mi.set_variant('cuda_ad_rgb_polarized')
xml_path = '/home/ctaglione/code/GraspNeRF/test/'
image_path = '/home/ctaglione/code/GraspNeRF/pola/'
if not os.path.exists(image_path):
    os.makedirs(image_path)

def register_image(xml_folder, output_folder):
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith(".xml") and "dict" in xml_file:
            xml_path = os.path.join(xml_folder, xml_file)
            mitsuba_render(xml_path, output_folder, xml_file)


def normalize_image(img):
    normalized_img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return normalized_img


def plot_stokes_component(ax, image):
    data = mi.TensorXf(image)[:, :, 1]
    plot_minmax = 0.05 * max(dr.max(data)[0], dr.max(-data)[0]) # Arbitrary scale for colormapy
    img = ax.imshow(data, cmap='coolwarm', vmin=-plot_minmax, vmax=+plot_minmax)
    ax.set_xticks([]); ax.set_yticks([])
    return img


def calculate_polarization_parameters(channels): #TODO: verify function
    I = mi.TensorXf(channels['S0'])
    Q = mi.TensorXf(channels['S1'])
    U = mi.TensorXf(channels['S2'])
    V = mi.TensorXf(channels['S3'])
    I = np.nan_to_num(I, nan=1e-10)
    DoP = np.sqrt(Q**2 + U**2 + V**2)
    np.divide(DoP, I, out=DoP, where=I != 0, casting='unsafe')
    AoP = np.mod(0.5 * np.arctan2(U,Q), np.pi)
    return DoP, AoP



def applyColorToAoLP(aolp: np.ndarray, saturation= 1.0, value= 1.0) -> np.ndarray:
    ones = np.ones_like(aolp)

    hue = (np.mod(aolp, np.pi) / np.pi * 179).astype(np.uint8)  # [0, pi] to [0, 179]
    saturation = np.clip(ones * saturation * 255, 0, 255).astype(np.uint8)
    value = np.clip(ones * value * 255, 0, 255).astype(np.uint8)

    hsv = cv2.merge([hue, saturation, value])
    aolp_colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return aolp_colored


def visualize_dop(dop):
    dop = cv2.cvtColor(dop, cv2.COLOR_BGR2GRAY)
    min_pixel_value = np.min(dop)
    max_pixel_value = np.max(dop)
    normalized_dop = 255 * (dop - min_pixel_value) / (max_pixel_value - min_pixel_value)
    normalized_dop = normalized_dop.astype(np.uint8)
    return normalized_dop



def mitsuba_render(xml_path, output_folder, xml_file):
    print('xml_path', xml_path)
    #scene = mi.load_file('/home/ctaglione/code/GraspNeRF/urdf/dict_0_0.xml')
    scene = mi.load_file(xml_path)
    params = mi.traverse(scene)
    #print(params)
    image = mi.render(scene, spp=512)
    bitmap = mi.Bitmap(image, channel_names=['R', 'G', 'B'] + scene.integrator().aov_names())
    #bitmap.write('dict.exr')
    channels = dict(bitmap.split())

    #import ipdb; ipdb.set_trace()
    # Plot parameters S0
    # plt.figure(figsize=(5, 5))
    # plt.imshow(channels['S0'].convert(srgb_gamma=True), cmap='gray')
    # plt.colorbar()
    # plt.xticks([]); plt.yticks([])
    # plt.xlabel("S0: Intensity", size=14, weight='bold')
    # plt.show()
    
    # # Plot parameters S1, S2, S3
    # fig, ax = plt.subplots(ncols=3, figsize=(18, 5))
    # img = plot_stokes_component(ax[0], channels['S1'])
    # plt.colorbar(img, ax=ax[0])
    # img = plot_stokes_component(ax[1], channels['S2'])
    # plt.colorbar(img, ax=ax[1])
    # img = plot_stokes_component(ax[2], channels['S3'])
    # plt.colorbar(img, ax=ax[2])
    # ax[0].set_xlabel("S1: Horizontal vs. vertical", size=14, weight='bold')
    # ax[1].set_xlabel("S2: Diagonal", size=14, weight='bold')
    # ax[2].set_xlabel("S3: Circular", size=14, weight='bold')
    # plt.show()
    

    degree_of_polarization, angle_of_polarization = calculate_polarization_parameters(channels)
    normalized_aop = applyColorToAoLP(angle_of_polarization[:,:,0])
    normalized_dop = visualize_dop(degree_of_polarization)
    # plt.imshow(normalized_dop, cmap='gray')
    # plt.title('Degree of Polarization (DoP)')
    # plt.axis('off')
    # plt.show()

    # plt.imshow(normalized_aop)
    # plt.title('Angle of Polarization Image')
    # plt.colorbar(label='Angle (degrees)')
    # plt.show()

    
    #Register all image 
    AoP_path = output_folder+ 'AoP/'
    DoP_path = output_folder + 'DoP/'
    S_path = output_folder + 'S/'
    if not os.path.exists(AoP_path):
         os.makedirs(AoP_path)
    if not os.path.exists(DoP_path):
         os.makedirs(DoP_path)
    if not os.path.exists(S_path):
        os.makedirs(S_path)

    image_name = os.path.splitext(os.path.basename(xml_path))[0]
    AoP_path = os.path.join(AoP_path,image_name+'AoP'+'.png')
    DoP_path = os.path.join(DoP_path,image_name+'DoP'+'.png')
    S0_path = os.path.join(S_path,image_name+'S0'+'.png')
    #S1_path = os.path.join(S_path,image_name, 'S1', '.png')
    #S2_path = os.path.join(S_path,image_name, 'S2', '.png')
    #S3_path = os.path.join(S_path,image_name, 'S3', '.png')
    print(DoP_path)
    print(AoP_path)
    print(S0_path)
    cv2.imwrite(AoP_path, normalized_aop)
    cv2.imwrite(DoP_path, normalized_dop)
    mi.util.write_bitmap(S0_path, mi.TensorXf(channels['S0']))


    #cv2.imwrite(S1_path, mi.TensorXf(channels['S1']))
    #cv2.imwrite(S2_path, mi.TensorXf(channels['S2']))
    #cv2.imwrite(S3_path, mi.TensorXf(channels['S3']))

register_image(xml_path, image_path)



