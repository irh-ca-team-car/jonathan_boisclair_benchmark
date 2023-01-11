const fs = (await import('fs'));
const fsp = fs.promises;
const path = await import('path');
import padLeft from "pad-left";

var annotations = await fsp.readFile("/media/boiscljo/LinuxData/Datasets/FLIR ADK FREE/FLIR_ADAS_1_3/train/thermal_annotations.json", "utf8")
var annotationsVal = await fsp.readFile("/media/boiscljo/LinuxData/Datasets/FLIR ADK FREE/FLIR_ADAS_1_3/val/thermal_annotations.json", "utf8")

annotations = JSON.parse(annotations);
annotationsVal = JSON.parse(annotationsVal);

function groupBy(objectArray, property) {
    return objectArray.reduce((acc, obj) => {
        const key = obj[property];
        if (!acc[key]) {
            acc[key] = [];
        }
        // Add object to list for given key's value
        acc[key].push(obj);
        return acc;
    }, {});
}

var data = groupBy(annotations.annotations, "image_id")
var dataVal = groupBy(annotationsVal.annotations, "image_id")

data = Object.keys(data).map(x => { return { category: x, data: data[x], quantity: data[x].length } })
dataVal = Object.keys(dataVal).map(x => { return { category: x, data: dataVal[x], quantity: dataVal[x].length } })

//console.log(dataVal);
//console.log(data)

var FLIR_DIR = "data/FLIR_CONVERTED"
var IMAGES_DIR = path.join(FLIR_DIR, "images")
var LABELS_DIR = path.join(FLIR_DIR, "labels")


async function delFile(delPath) {
    try {
        var files = await fsp.readdir(delPath)
        await Promise.all(files.map(file => delFile(path.join(delPath, file))));
    } catch {

    }
    await fsp.rm(delPath, { recursive: true });
}

async function doElement(element) {
    if (fs.existsSync(element))
        await delFile(element);
    if (!fs.existsSync(element)) {
        await fsp.mkdir(element);
    }
}

await doElement(FLIR_DIR);
await doElement(IMAGES_DIR);
await doElement(LABELS_DIR);

function imagePath(id) {
    var img = annotations.images.filter(x => x.id == id)[0]

    return {
        jpg: img.file_name.replace("thermal_8_bit", "RGB").replace(".jpeg", ".jpg"),
        tiff: img.file_name.replace("thermal_8_bit", "thermal_16_bit").replace(".jpeg", ".tiff"),
        img_data: img
    }
}
function imagePathVal(id) {
    var img = annotationsVal.images.filter(x => x.id == id)[0]

    
    return {
        jpg: img.file_name.replace("thermal_8_bit", "RGB").replace(".jpeg", ".jpg"),
        tiff: fs.existsSync(img.file_name)? img.file_name: img.file_name.replace("thermal_8_bit", "thermal_16_bit").replace(".jpeg", ".tiff"),
        img_data: img
    }
}
await fsp.writeFile(path.join(FLIR_DIR, "all.csv"), "");
await fsp.writeFile(path.join(FLIR_DIR, "val.csv"), "");

await Promise.all(Object.values(data).map(async element => {
    //console.log(element);

    var { jpg, tiff, img_data } = imagePath(element.category)

    var num = padLeft(element.category, 5, "0");
    var train_dir = path.resolve("/media/boiscljo/LinuxData/Datasets/FLIR ADK FREE/FLIR_ADAS_1_3/train/");
    var jpg = path.join(train_dir, jpg)
    var tiff = path.join(train_dir, tiff)
    try {
        await fsp.access(jpg)
        await fsp.access(tiff)
        await fsp.copyFile(jpg, path.join(IMAGES_DIR, num + ".jpg"));
        await fsp.copyFile(tiff, path.join(IMAGES_DIR, num + ".tiff"));
        await fsp.appendFile(path.join(FLIR_DIR, "all.csv"), "3:images/" + num + ".jpg,1:images/" + num + ".tiff,labels/" + num + ".txt\r\n");

        var w = img_data.width;
        var h = img_data.height;

        await Promise.all(element.data.map(async x => {

            var bbox = x.bbox;
            bbox[0] /= w;
            bbox[1] /= h;
            bbox[2] /= w;
            bbox[3] /= h;
            await fsp.appendFile(path.join(LABELS_DIR, num + ".txt"), x.category_id + " " + x.bbox.join(" ") + "\r\n")
        }))
    } catch (ex) {
        console.error("error",ex);
    }

}))

await Promise.all(Object.values(dataVal).map(async element => {
    //console.log(element);

    var { jpg, tiff, img_data } = imagePathVal(element.category)

    var num = padLeft(element.category, 5, "0");
    var train_dir = path.resolve("/media/boiscljo/LinuxData/Datasets/FLIR ADK FREE/FLIR_ADAS_1_3/val/");
    var jpg = path.join(train_dir, jpg)
    var tiff = path.join(train_dir, tiff)
    try {
        await fsp.access(jpg)
        await fsp.access(tiff)
        await fsp.copyFile(jpg, path.join(IMAGES_DIR, num + ".jpg"));
        await fsp.copyFile(tiff, path.join(IMAGES_DIR, num + ".tiff"));
        await fsp.appendFile(path.join(FLIR_DIR, "val.csv"), "3:images/" + num + ".jpg,1:images/" + num + ".tiff,labels/" + num + ".txt\r\n");

        var w = img_data.width;
        var h = img_data.height;

        await Promise.all(element.data.map(async x => {

            var bbox = x.bbox;
            bbox[0] /= w;
            bbox[1] /= h;
            bbox[2] /= w;
            bbox[3] /= h;
            await fsp.appendFile(path.join(LABELS_DIR, num + ".txt"), x.category_id + " " + x.bbox.join(" ") + "\r\n")
        }))
    } catch (ex){
        console.error("error",ex);
    }

}))
