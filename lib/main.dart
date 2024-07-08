import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show Uint8List, rootBundle;
import 'package:image/image.dart' as img;
import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import 'package:tflite_flutter/tflite_flutter.dart';

void main() => runApp(const MyApp());

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  Image? _image;
  Uint8List? _output;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  // Load the TFLite model
  Future<void> _loadModel() async {
    const String model3Name = 'assets/LCC_Tflite_model_ShapeChanged.tflite';
    const String imagePath = 'assets/1.png';

    final interpreter = await Interpreter.fromAsset(model3Name);
    interpreter.allocateTensors();
    final inputShape = interpreter.getInputTensor(0).shape;
    final outputShape = interpreter.getOutputTensor(0).shape;
    print(inputShape);
    print(outputShape);

    final image = await loadImage(imagePath);
    img.Image preprocessedImage = preprocessImage(image!);
    setState(() {
      _image =
          Image.memory(Uint8List.fromList(img.encodePng(preprocessedImage)));
    });

    // Prepare the input tensor
    var input = imageToTensor(preprocessedImage, 256, 256);
    // Allocate space for output tensor
    var output = List.generate(
        1,
        (_) => List.generate(
            256, (_) => List.generate(256, (_) => List.filled(1, 0.0))));
    // Run inference
    interpreter.run(input, output);

    final maskImage = applyMaskToImage(preprocessedImage, output);
    setState(() {
      _output = maskImage;
    });

    // Close the interpreter
    interpreter.close();
  }

  Uint8List applyMaskToImage(
      img.Image originalImage, List<List<List<List<double>>>> tensor) {
    int width = originalImage.width;
    int height = originalImage.height;

    // Assuming the tensor's dimensions match the image's dimensions
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        // Determine if the current pixel is part of the foreground
        bool isForeground = tensor[0][y][x][0] > 0.5;

        // If the pixel is not part of the foreground, set its alpha to 0
        if (!isForeground) {
          originalImage.setPixelRgba(
              x, y, 0, 0, 0, 0); // Set pixel to fully transparent
        }
      }
    }

    // Encode the modified image to PNG and return as Uint8List
    return Uint8List.fromList(img.encodePng(originalImage));
  }

  Future<Uint8List?> removeBackground(
      Uint8List originalImage, Uint8List maskImage) async {
    return null;
  }

  Future<Uint8List?> _convertUiImageToByteData(ui.Image image) async {
    final ByteData? byteData =
        await image.toByteData(format: ui.ImageByteFormat.png);

    return byteData?.buffer.asUint8List();
  }

  // Prints true if there's at least one 1.0 in the output, otherwise false

  Future<img.Image?> loadImage(String assetPath) async {
    final ByteData data = await rootBundle.load(assetPath);
    List<int> bytes = data.buffer.asUint8List();
    img.Image? image = img.decodeImage(Uint8List.fromList(bytes));
    return image;
  }

  img.Image preprocessImage(img.Image image) {
    // Resize the image to 256x256
    img.Image resized = img.copyResize(image, width: 256, height: 256);
    return resized;
  }

  List<List<List<List<double>>>> imageToTensor(
      img.Image image, int width, int height) {
    // Initialize the tensor as a 1x256x256x3 list (initialized to zeroes)
    var tensor = List.generate(
        1,
        (_) => List.generate(
            height, (_) => List.generate(width, (_) => List.filled(3, 0.0))));

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        var pixel = image.getPixel(x, y);
        tensor[0][y][x][0] = (img.getRed(pixel) / 255.0); // Normalize Red
        tensor[0][y][x][1] = (img.getGreen(pixel) / 255.0); // Normalize Green
        tensor[0][y][x][2] = (img.getBlue(pixel) / 255.0); // Normalize Blue
      }
    }

    return tensor;
  }
  // Run tflite inference on the provided image

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('TFLite Model Example'),
        ),
        body: Column(
          children: <Widget>[
            // ignore: unnecessary_null_comparison
            (_image != null) ? _image! : Container(),
            (_output != null) ? Image.memory(_output!) : Container(),
          ],
        ),
        floatingActionButton: FloatingActionButton(
          onPressed: () {},
          tooltip: 'Pick Image',
          child: const Icon(Icons.image),
        ),
      ),
    );
  }
}
