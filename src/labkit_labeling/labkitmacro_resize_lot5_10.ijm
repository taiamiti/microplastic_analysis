//setBatchMode(true);
//inputDir = getDirectory("Choose image directory ");
//modelDir = getDirectory("Choose models directory ");
//outputDir = getDirectory("Choose Output image directory ");
inputDir = "/home/taiamiti/Projects/micro-plastic/cmdinferencelabkit/reannotation/Photos annotation/lot5_10/";
modelDir = "/home/taiamiti/Projects/micro-plastic/cmdinferencelabkit/reannotation/Model_Irene/lot5_10/";
outputDir = "/home/taiamiti/Projects/micro-plastic/cmdinferencelabkit/output_reannot_lot5-10/";

processRoot(inputDir, modelDir, outputDir);



function contains( array, value ) {
    for (i=0; i<array.length; i++) 
        if ( array[i] == value ) return true;
    return false;
}


function processRoot(inputDir, modelDir, outputDir) {
	models = getFileList(modelDir);
	models = Array.sort(models);
	for (i = 0; i < models.length; i++) {
		model_path = modelDir + models[i];
		task_id = replace(models[i], ".classifier", "");  // suppose model are named [task_id].classifier
		input_folder = inputDir + task_id;
		output_folder = outputDir + task_id;
		print(input_folder);
		processFolder(input_folder, output_folder, model_path);
	}
}

function processFolder(inputDir, outputDir, model_path) {
	print(model_path);
	imgs = getFileList(inputDir);
	outputs = getFileList(outputDir);
	for (j = 0; j < imgs.length; j++) {
		if (contains(outputs, replace(imgs[j], ".jpg", ".png")) == false) {
			processImage(inputDir, model_path, imgs[j]);
		}
	}
}


function processImage(sel_dir, model_path, imageFile) {
	img_path = sel_dir + File.separator + imageFile;
	print(img_path);
	open(img_path);
	run("Scale...", "x=0.5 y=0.5 width=960 height=600 interpolation=Bilinear average create");
	run("Segment Image With Labkit", "segmenter_file=" + model_path + " use_gpu=False");
	output_path = outputDir + File.separator + replace(imageFile, ".jpg", ".png");
	print(output_path);
	File.makeDirectory(File.getParent(output_path));
	run("Multiply...", "value=255");
	//run("8-bit");
	run("Scale...", "x=2 y=2 width=190 height=100 interpolation=None average create");
	//setTool("rectangle");
	setBackgroundColor(0, 0, 0);
	wait(100);
	makeRectangle(1, 1149, 220, 46); //remove left text (zoom info)
	wait(500);
	run("Clear", "slice");
	makeRectangle(1680, 1155, 236, 42); //remove right text (zoom info)
	wait(500);
	run("Clear", "slice");
	run("Gray Morphology", "radius=10 type=circle operator=close");
	saveAs("PNG", output_path);
	close("*");  // close all images
	wait(100);
}
