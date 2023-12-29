import xml.etree.ElementTree as ET 
import os

xml_path = '/home/ctaglione/code/GraspNeRF/test/'
image_path = '/home/ctaglione/code/GraspNeRF/xml_rgb/'
if not os.path.exists(image_path):
    os.makedirs(image_path)

def register_image(xml_folder, output_folder):
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith(".xml") and "dict" in xml_file:
            xml_path = os.path.join(xml_folder, xml_file)
            modify_integrator(xml_path, output_folder)

def modify_integrator(xml_file_path, output_folder):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    integrator_element = root.find(".//integrator")
    root.remove(integrator_element)
    new_integrator = ET.Element('integrator', type='path')
    rgb_element = ET.SubElement(new_integrator, 'integer', name='max_depth', value='-1')
    root.append(new_integrator)
    image_name = os.path.splitext(os.path.basename(xml_file_path))
    print(image_name)
    new_path = os.path.join(output_folder,image_name[0]+'rgb'+image_name[1] )
    tree.write(new_path)
    print(f"XML file saved: {new_path}")

register_image(xml_path, image_path)



