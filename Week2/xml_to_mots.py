import xml.etree.ElementTree as ET

def read_ground_truth(xml_file, classes):
    '''
    [
        [[x1, y1, x2, y2], [x1, y1, x2, y2], …],
        [[x1, y1, x2, y2], [x1, y1, x2, y2], …], 
        …,
        [[...]]
    ]
    '''
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # bbox_info = [[] for _ in range(n_frames)]

    # print(len(bbox_info))
    dict_frames = {}

    for track in root.findall('./track'):
        label = track.attrib['label']
        id = int(track.attrib['id'])

        if label in classes:  #Classes contains ['bike', 'car']
            for box in track.findall('box'):
                parked = False
                for attribute in box.findall('attribute'):
                    if attribute.attrib.get('name') == 'parked' and attribute.text == 'true':
                        parked = True
                        break
                if not parked:
                    frame = int(box.attrib['frame'])
                    if frame not in dict_frames:
                        dict_frames[frame] = []
                    xtl = int(float(box.attrib['xtl']))
                    ytl = int(float(box.attrib['ytl']))
                    xbr = int(float(box.attrib['xbr']))
                    ybr = int(float(box.attrib['ybr']))
                    w = xbr-xtl
                    h = ybr-ytl
                    conf = 1

                    dict_frames[frame].append([frame, id, xtl,ytl, w,h,conf, -1,-1,-1])
                    
        
    return dict_frames

if __name__ == "__main__":

    xml_path = '/ghome/group02/C6/Week1/ai_challenge_s03_c010-full_annotation.xml'
    output_txt_path = 'gt_s03_C010.txt'

    dict_out = read_ground_truth(xml_path, ['bike', 'car'])

    #Order frames in ascending order
    ordered_dict = dict(sorted(dict_out.items()))

    with open(output_txt_path, 'w') as file:
        for frame, values in ordered_dict.items():
            for value in values:
                file.write(f"{str(value).strip('[]')}\n")