import os
import numpy as np
import mitsuba as mi
import matplotlib.pyplot as plt

mi.set_variant('cuda_ad_rgb')

xml_path = '/home/ctaglione/code/GraspNeRF/xml_rgb/'
image_path = '/home/ctaglione/code/GraspNeRF/pola/rgb/'
if not os.path.exists(image_path):
    os.makedirs(image_path)

def register_image(xml_folder, output_folder):
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith(".xml") and "dict" in xml_file:
            xml_path = os.path.join(xml_folder, xml_file)

            mitsuba_render_doc(xml_path, output_folder)


# def mitsuba_render_github(xml_path, output_folder, xml_file):
#     scene = mi.load_file('../scenes/simple.xml')
#     #scene = mi.load_file(xml_path)
#     scene.integrator().render(scene, scene.sensors()[0])
#     film = scene.sensors()[0].film()


#     # # Add the scene directory to the FileResolver's search path
#     # Thread.thread().file_resolver().append(os.path.dirname(filename))
    
#     # After rendering, the rendered data is stored in the film
#     film = scene.sensors()[0].film()

#     # Write out rendering as high dynamic range OpenEXR file
#     film.set_destination_file(output_folder)
#     film.develop()

#     # Write out a tonemapped JPG of the same rendering
#     bmp = film.bitmap(raw=True)
#     bmp.convert(mi.core.Bitmap.PixelFormat.RGB, mi.core.Struct.Type.UInt8, srgb_gamma=True).write('/path/to/output.jpg')

#     # Get linear pixel values as a numpy array for further processing
#     bmp_linear_rgb = bmp.convert(mi.core.Bitmap.PixelFormat.RGB, mi.core.Struct.Type.Float32, srgb_gamma=False)
#     image_np = np.array(bmp_linear_rgb)
#     plt.axis("off")
#     plt.imshow(image_np); 

    


def mitsuba_render_doc(xml_path, output_folder): 
    print(xml_path)
    scene = mi.load_file(xml_path)
    image = (mi.render(scene, spp=256))** (1.0 / 2.2)
    print(image)
    #plt.axis("off")
    #plt.imshow(image*10)
    #plt.imshow(image ** (1.0 / 2.2)); # approximate sRGB tonemapping
    #plt.show()

    image_name = os.path.splitext(os.path.basename(xml_path))[0]
    rgb_path = os.path.join(output_folder,image_name + '.png')
    #mi.util.write_bitmap(rgb_path, image)





register_image(xml_path, image_path)