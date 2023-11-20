from owslib.wms import WebMapService
import os
import matplotlib.pyplot as plt
import PIL.Image as PImage

wms = WebMapService('https://geo2.dgterritorio.gov.pt/geoserver/COS2015/wms', version='1.3.0')

print(wms.identification.type)
print(wms.identification.version)
print(wms.identification.title)
print(wms.identification.abstract)
print(list(wms.contents))
print(wms['COS2015v2'].title)
print(wms['COS2015v2'].queryable)
print(wms['COS2015v2'].opaque)
print(wms['COS2015v2'].boundingBox)
print(wms['COS2015v2'].boundingBoxWGS84)
print(wms['COS2015v2'].crsOptions)
#['EPSG:900913', 'EPSG:4258', 'EPSG:3763', 'EPSG:4326', 'EPSG:3857']
print(wms['COS2015v2'].styles)
#{
# 'inspire_common:DEFAULT':
#   {'title': 'inspire_common:DEFAULT', 'legend': 'http://mapas.dgterritorio.pt/wms-inspire/cos2015v1?language=por&version=1.3.0&service=WMS&request=GetLegendGraphic&sld_version=1.1.0&layer=COS2015v1&format=image/png&STYLE=inspire_common:DEFAULT', 'legend_width': '395', 'legend_height': '838', 'legend_format': 'image/png'},
# 'default':
#   {'title': 'default', 'legend': 'http://mapas.dgterritorio.pt/wms-inspire/cos2015v1?language=por&version=1.3.0&service=WMS&request=GetLegendGraphic&sld_version=1.1.0&layer=COS2015v1.0&format=image/png&STYLE=default', 'legend_width': '395', 'legend_height': '821', 'legend_format': 'image/png'}
#}

print([op.name for op in wms.operations])
print(wms.getOperationByName('GetMap').methods)
#[
# {'type': 'Get', 'url': 'http://mapas.dgterritorio.pt/wms-inspire/cos2015v1?language=por&'},
# {'type': 'Post', 'url': 'http://mapas.dgterritorio.pt/wms-inspire/cos2015v1?language=por&'}
#]
print(wms.getOperationByName('GetMap').formatOptions)
# ['image/png', 'image/jpeg', 'image/gif', 'image/png; mode=8bit', 'application/x-pdf', 'image/svg+xml', 'image/tiff', 'application/vnd.google-earth.kml+xml', 'application/vnd.google-earth.kmz']

img = wms.getmap(
    layers=['COS2015v2'],
    # styles=['default'],
    srs='EPSG:3857',
    bbox=(-961272.067714374, 5090094.587566513, -958826.082809240, 5092540.572471580),
    # bbox=(-1061557.4488245286047459, 4432124.6480876598507166, -687321.7583403056487441, 5185487.9988663569092751),
    size=(256, 256),
    # size=(2600, 2600),
    format='image/tiff',
    # format='image/png; mode=8bit',
    transparent=True
    )
folder = "../Files"
file_name = 'COS_tile.tif'
out = open(os.path.join(folder, file_name), 'wb')
# out = open('COS_tile.png', 'wb')
dados = img.read()
out.write(dados)
out.close()

image = PImage.open(os.path.join(folder, file_name))
plt.imshow(image)
plt.show()
