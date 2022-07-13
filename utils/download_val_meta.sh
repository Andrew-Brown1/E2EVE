mkdir ../E2EVE_meta/
mkdir ../E2EVE_output/
mkdir ../E2EVE_output/logs/

wget -P ../E2EVE_meta/ https://www.robots.ox.ac.uk/~abrown/E2EVE/resources/FFHQ_meta.zip

unzip ../E2EVE_meta/FFHQ_meta.zip -d ../E2EVE_meta/
mv ../E2EVE_meta/FFHQ_meta/vgg.pth ../E2EVE_meta/
mv ../E2EVE_meta/FFHQ_meta/pt_inception-2015-12-05-6726825d.pth ../E2EVE_meta/

rm ../E2EVE_meta/FFHQ_meta.zip
