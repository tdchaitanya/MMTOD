def getXMLAnn(ann):
    st=""
    for i,b in enumerate(ann):
        if i == 0:
            st += f'\
        <object>\n \
                <name>person</name>\n \
                <pose>Frontal</pose>\n \
                <truncated>0</truncated>\n \
                <difficult>0</difficult>\n \
                <occluded>0</occluded>\n \
                <bndbox>\n \
                    <xmin>{b[0]}</xmin>\n \
                    <xmax>{int(b[0])+int(b[2])}</xmax>\n \
                    <ymin>{b[1]}</ymin>\n \
                    <ymax>{int(b[1])+int(b[3])}</ymax>\n \
                </bndbox>\n \
            </object> \n'
        else:
            st += f'\
            <object>\n \
                <name>person</name>\n \
                <pose>Frontal</pose>\n \
                <truncated>0</truncated>\n \
                <difficult>0</difficult>\n \
                <occluded>0</occluded>\n \
                <bndbox>\n \
                    <xmin>{b[0]}</xmin>\n \
                    <xmax>{int(b[0])+int(b[2])}</xmax>\n \
                    <ymin>{b[1]}</ymin>\n \
                    <ymax>{int(b[1])+int(b[3])}</ymax>\n \
                </bndbox>\n \
            </object> \n'
    return st

def toXML(name,ann):

    x = f'\
<annotation>\n \
    <folder>/DATA1/chaitanya/exp/MultiModal/Experiment/images/lwir</folder>\n \
    <filename>{name.replace(".txt",".jpg")}</filename>\n \
    <path>/DATA1/chaitanya/exp/MultiModal/Experiment/images/lwir/{name.replace(".txt",".jpg")}</path>\n \
    <source>\n \
        <database>KAIST lwir</database>\n \
    </source>\n \
    <size>\n \
        <width>640</width>\n \
        <height>512</height>\n \
        <depth>3</depth>\n \
    </size>\n \
    <segmented>0</segmented>\n \
    {getXMLAnn(ann)} \
</annotation>\n'
    return x
