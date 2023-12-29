import xml.etree.ElementTree as ET
import os

xml_path = '/home/ctaglione/code/GraspNeRF/xml/'
xml_path_2 = '/home/ctaglione/code/GraspNeRF/test/'
if not os.path.exists(xml_path_2):
    os.makedirs(xml_path_2)


def remove_shape_by_condition(xml_folder,xml_folder_2):
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith(".xml") and "dict" in xml_file:
            xml_path2 = os.path.join(xml_folder_2, xml_file)
            xml_path = os.path.join(xml_folder, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for shape in root.findall(".//shape"):
                string_element = shape.find('.//string[@name="filename"]')
                for attrib_name, attrib_value in string_element.attrib.items():
                    print(f"{attrib_name}: {attrib_value}")
                if string_element is not None and 'path.obj' in string_element.get('value', ''):
                    root.remove(shape)
                    print('remove')
                else:
                    print("No 'path.obj' found in any 'value' attribute.")
            tree.write(xml_path2)





remove_shape_by_condition(xml_path, xml_path_2)