"""example usage usage "/home/mohammed-alaa/Downloads/Mock Claims 6-v2.pdf" --is_complex True --specific_page 3 --compress_images True"""
"""
versions..

magick:
Version: ImageMagick 7.0.8-15 Q16 x86_64 2018-12-12 https://imagemagick.org
Copyright: © 1999-2018 ImageMagick Studio LLC
License: https://imagemagick.org/script/license.php
Features: Cipher DPC HDRI OpenMP 
Delegates (built-in): fontconfig freetype heic jbig jng jp2 jpeg lzma pangocairo png tiff webp x zlib

tesseract:
tesseract 4.0.0-beta.1
 leptonica-1.75.3
  libgif 5.1.4 : libjpeg 8d (libjpeg-turbo 1.5.2) : libpng 1.6.34 : libtiff 4.0.9 : zlib 1.2.11 : libwebp 0.6.1 : libopenjp2 2.3.0

 Found AVX
 Found SSE
"""
import argparse
import glob
import os

params_parser = argparse.ArgumentParser(description='Searchable PDF script')

params_parser.register('type', 'bool', lambda v: v.lower() in ('yes', 'true', 't', '1', 'y'))
params_parser.add_argument('source')
params_parser.add_argument('--is_complex', type='bool', default=True, dest="complex processing by magick")
params_parser.add_argument('--specific_page', type=int, default=-1, dest="select a page or it loops on the entire document(better if you select- it's zero based)")
params_parser.add_argument('--compress_images', type='bool', default=True, dest="lossy compression of the image by opencv(preferred)")

params = params_parser.parse_args()

use_imageMagick = True
imageMagick_source = "/home/mohammed-alaa/Source/imagemagick/ImageMagick-7.0.8-15/utilities/magick"  # "convert

source = params.source
is_complex = params.is_complex
specific_page = params.specific_page
compress_images = params.compress_images

is_pdf = os.path.split(source)[1].split(".")[1] == "pdf"
if is_pdf:
    print("source_pdf", source)
else:
    print("source image", source)

print("is_complex", is_complex)
print("compress_images", compress_images)

if is_pdf:
    print("specific_page", specific_page)
else:
    print("ignoring specific_page for image input")

if is_pdf:
    os.system("{} -density 10 '{}' '{}'".format(imageMagick_source, source, os.path.join(os.path.split(source)[0], "image.jpg")))
    generated_images = glob.glob(os.path.join(os.path.split(source)[0], "image*.jpg"))

    if len(generated_images) > 1:
        generated_images = sorted(generated_images, key=lambda item: int(os.path.split(item)[1].replace("image-", "").replace(".jpg", "")))

    if specific_page != -1:
        generated_images = [generated_images[specific_page]]

    for i, small_image in enumerate(generated_images):
        if len(os.path.split(small_image)[-1].split("-")) > 1:
            index = int(os.path.split(small_image)[-1].split("-")[-1].split(".")[0])
        else:
            index = 0
        os.rename(small_image, source.split(".")[0] + "_{}.tiff".format(index))  # output tiff
        generated_images[i] = source.split(".")[0] + "_{}.tiff".format(index)  # output tiff

    for output_file in generated_images:
        page = os.path.split(output_file)[1].split(".")[0].split("_")[-1]
        # if not is_complex:
        #     output_file=output_file.replace("tiff","jpg")
        os.system("{} -density 300 '{}'[{}] {} '{}'".format(imageMagick_source, source, page, "-resize 250% -depth 8 -strip -background white -flatten +matte -alpha off -negate -lat 25x25+10% -negate" if is_complex else "", output_file))
        print("{} -density 300 '{}'[{}] {} '{}'".format(imageMagick_source, source, page, "-resize 250% -depth 8 -strip -background white -flatten +matte -alpha off -negate -lat 25x25+10% -negate" if is_complex else "", output_file))

        print("generated:" + output_file)

    print("made separate images")
else:
    output_file = os.path.join(os.path.split(source)[0], os.path.split(source)[1].split(".")[0] + "-magick." + os.path.split(source)[1].split(".")[1])
    if use_imageMagick:
        # if not is_complex:
        #     output_file=output_file.replace("tiff","jpg")
        # -flatten +matte
        os.system("{} -density 300 '{}' {} '{}'".format(imageMagick_source, source, "-resize 250% -depth 8 -strip -background white  -alpha off -negate -lat 25x25+10% -negate" if is_complex else "", output_file))
        print("{} -density 300 '{}' {} '{}'".format(imageMagick_source, source, "-resize 250% -depth 8 -strip -background white  -alpha off -negate -lat 25x25+10% -negate" if is_complex else "", output_file))
    else:
        from shutil import copyfile

        copyfile(source, output_file)
        print("copying into", output_file)

    print("generated:" + output_file)
    generated_images = [output_file]

with open("/tmp/filelist", "w") as f:
    f.writelines("\n".join(generated_images))

with open("/tmp/conf", "w") as f:
    f.writelines("tessedit_pageseg_mode 6\ntessedit_write_images 1\ntessedit_create_hocr 1")  # tessedit_create_pdf 1\n

print("made file list")
dest = source.split(".")[0] + ("-pdfinput" if is_pdf else "-imageinput") + "-sandwich{}".format("" if specific_page == -1 or not is_pdf else "-" + str(specific_page))
os.system("tesseract /tmp/filelist '{}' /tmp/conf".format(dest))

print("tesseract /tmp/filelist '{}' /tmp/conf".format(dest))

print("generated:" + dest)
if compress_images:
    import cv2

    print("compressing images..")
    for image_path in generated_images:
        image_ = cv2.imread(image_path)
        cv2.imwrite(image_path, image_)

for image in glob.glob(os.path.join(os.path.split(source)[0], "image*.jpg")):
    os.remove(image)
