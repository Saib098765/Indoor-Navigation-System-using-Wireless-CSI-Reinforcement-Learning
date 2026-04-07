import 'package:tflite_flutter/tflite_flutter.dart';

class AILocationEngine {
  late Interpreter _interpreter;
  bool isLoaded = false;

  Future<void> loadModel(String modelPath) async {
    try {
      _interpreter = await Interpreter.fromAsset(modelPath);
      isLoaded = true;
      print("AI Model Loaded: $modelPath");
    } catch (e) {
      print(" Failed: $e");
    }
  }

  int predictLocation(List<double> wifiFeatures) {
    if (!isLoaded) {
      print("Model not loaded");
      return -1;
    }

    var input = [wifiFeatures]; 
    var output = List.filled(1 * 748, 0.0).reshape([1, 748]);
    _interpreter.run(input, output);
    List<dynamic> probabilities = output[0];
    int predictedNodeId = 0;
    double maxProb = probabilities[0];

    for (int i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > maxProb) {
        maxProb = probabilities[i];
        predictedNodeId = i;
      }
    }
    
    return predictedNodeId;
  }
}