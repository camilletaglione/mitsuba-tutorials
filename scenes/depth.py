import os
import mitsuba as mi
import matplotlib.pyplot as plt

mi.set_variant('cuda_ad_rgb')

xml_path = '/home/ctaglione/code/GraspNeRF/xml_depth/'
image_path = '/home/ctaglione/code/GraspNeRF/pola/depth/'
if not os.path.exists(image_path):
    os.makedirs(image_path)

def register_image(xml_folder, output_folder):
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith(".xml") and "dict" in xml_file:
            xml_path = os.path.join(xml_folder, xml_file)

            mitsuba_render(xml_path, output_folder)



def mitsuba_render(xml_path, output_folder):
    print(xml_path)
    scene = mi.load_file(xml_path)
    image = mi.render(scene, spp=256)
    # plt.axis("off")
    # plt.imshow(image)
    # plt.show()
    image_name = os.path.splitext(os.path.basename(xml_path))[0]
    depth_path = os.path.join(output_folder,image_name + '.png')
    mi.util.write_bitmap(depth_path, image)


register_image(xml_path, image_path)