import numpy as np
import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
import cv2 as cv


mi.set_variant('cuda_ad_rgb_polarized')

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
    DoP = np.sqrt(Q**2 + U**2 + V**2) / I
    AoP = np.mod(0.5 * np.arctan2(U,Q), np.pi)
    return DoP, AoP

def applyColorToAoLP(aolp: np.ndarray, saturation= 1.0, value= 1.0) -> np.ndarray:
    ones = np.ones_like(aolp)

    hue = (np.mod(aolp, np.pi) / np.pi * 179).astype(np.uint8)  # [0, pi] to [0, 179]
    saturation = np.clip(ones * saturation * 255, 0, 255).astype(np.uint8)
    value = np.clip(ones * value * 255, 0, 255).astype(np.uint8)

    hsv = cv.merge([hue, saturation, value])
    aolp_colored = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return aolp_colored

#scene = mi.load_file('../scenes/cbox_pol.xml')
scene = mi.load_file('/home/ctaglione/code/GraspNeRF/xml/dict_0_0.xml')
image = mi.render(scene, spp=512)
params = mi.traverse(scene)
params = mi.traverse(scene)
bitmap = mi.Bitmap(image, channel_names=['R', 'G', 'B'] + scene.integrator().aov_names())
bitmap.write('dict.exr')
channels = dict(bitmap.split())

# # Plot parameters S0
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

_, angle_of_polarization = calculate_polarization_parameters(channels)

A = applyColorToAoLP(angle_of_polarization[:,:,0])
plt.imshow(A)
plt.show()




"""
degree_of_polarization, angle_of_polarization = calculate_polarization_parameters(channels)
dop = cv2.cvtColor(degree_of_polarization, cv2.COLOR_BGR2GRAY)

#plot_aop_with_colormap(angle_of_polarization)
#plot_dop_with_colormap(degree_of_polarization)

min_pixel_value = np.min(dop)
max_pixel_value = np.max(dop)

normalized_image = 255 * (dop - min_pixel_value) / (max_pixel_value - min_pixel_value)

normalized_image = normalized_image.astype(np.uint8)

plt.imshow(degree_of_polarization, cmap='gray')
plt.title('Degree of Polarization (DoP)')
plt.axis('off')
plt.show()

#return color
def visualize_aop(aop):
    #hsv_image = np.zeros_like(aop)
    hsv_image = cv2.cvtColor(aop,cv2.COLOR_BGR2HSV)
    hsv_image[:,:,0]=(hsv_image[:,:,0]%180).astype(np.uint8)
    hsv_image[:,:,1]=255
    hsv_image[:,:,2]=255
    rgb_image = cv2.cvtColor(aop,cv2.COLOR_HSV2BGR)
    return rgb_image

def visualisation_aop(aolp: np.ndarray, saturation: Union[float, np.ndarray] = 1.0, value: Union[float, np.ndarray] = 1.0) -> np.ndarray:
    ones = np.ones_like(aolp)
    hue = (np.mod(aolp, np.pi) / np.pi * 179).astype(np.uint8)  # [0, pi] to [0, 179]
    saturation = np.clip(ones * saturation * 255, 0, 255).astype(np.uint8)
    value = np.clip(ones * value * 255, 0, 255).astype(np.uint8)

    hsv = cv2.merge([hue, saturation, value])
    aolp_colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    min_pixel_value = np.min(aolp_colored)
    max_pixel_value = np.max(aolp_colored)
    normalized_aop = 255 * (aolp_colored - min_pixel_value) / (max_pixel_value - min_pixel_value)
    normalized_aop = normalized_aop.astype(np.uint8)
    white_threshold = 250 
    white_mask = (normalized_aop > white_threshold)
    normalized_aop[white_mask] = 0
    return normalized_aop

#aop_normalized = (aop_degrees % 360) / 360.0
#rgb_image = hsv_to_rgb(aop_normalized)
#aop = visualisation_aop(angle_of_polarization)
normalized_aop = visualize_aop(angle_of_polarization)

plt.imshow(normalized_aop)
plt.title('Angle of Polarization Image')
plt.colorbar(label='Angle (degrees)')
plt.show()"""











