import 'dart:async';
import 'dart:ffi';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show Uint8List, rootBundle;
import 'package:image/image.dart' as img;
import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import 'package:tflite_flutter/tflite_flutter.dart';

typedef Tensor4D = List<List<List<List<double>>>>;
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

    // Load model and image
    final interpreter = await Interpreter.fromAsset(model3Name);
    final image = await loadImage(imagePath);
    if (image == null) return;

    // Preprocess the image
    img.Image preprocessedImage = preprocessImage(image);
    Tensor4D input = imageToTensor(preprocessedImage, 256, 256);
    setState(() => _image =
        Image.memory(Uint8List.fromList(img.encodePng(preprocessedImage))));

    // Adjust the shape of the output tensor to match the interpreter's expected output
    Tensor4D output = getOutputTensor();

    // Run inference
    interpreter.run(input, output);

    // Post-process to create mask image
    final maskImage = applyMaskToImage(preprocessedImage, output);
    setState(() => _output = maskImage);

    // Close the interpreter
    interpreter.close();
  }

  Tensor4D getOutputTensor() {
    return List.generate(
      1,
      (_) => List.generate(
        256,
        (_) => List.generate(
          256,
          (_) => List.filled(1, 0.0),
        ),
      ),
    );
  }

  Uint8List applyMaskToImage(img.Image originalImage, Tensor4D tensor) {
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

  Tensor4D imageToTensor(img.Image image, int width, int height) {
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
        body: _output != null
            ? Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: <Widget>[
                    // ignore: unnecessary_null_comparison
                    (_image != null) ? _image! : Container(),
                    const SizedBox(height: 20),
                    (_output != null) ? Image.memory(_output!) : Container(),
                  ],
                ),
              )
            : const CircularProgressIndicator(),
        floatingActionButton: FloatingActionButton(
          onPressed: () {},
          tooltip: 'Pick Image',
          child: const Icon(Icons.image),
        ),
      ),
    );
  }
}
