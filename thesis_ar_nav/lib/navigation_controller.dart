import 'dart:convert';
import 'dart:math' as math;
import 'dart:collection';
import 'package:flutter/services.dart';
import 'package:wifi_scan/wifi_scan.dart';
import 'ai_engine.dart';

class NavigationController {
  final AILocationEngine aiEngine = AILocationEngine();
  Map<String, dynamic> mapGraph = {};
  
  int currentNodeId = -1;
  int targetNodeId = -1; 
  double targetBearing = 0.0;
  double distanceToTarget = 0.0; 
  int nodesRemaining = 0; 
  
  int currentFloor = 0;
  int nextFloor = 0;
  
  String scanStatus = "Initializing"; 
  bool isInitializing = true;

  Future<void> initialize(String modelPath, String mapPath) async {
    try {
      isInitializing = true;
      scanStatus = "Loading assets"; // Updates the UI so you know it started

      await aiEngine.loadModel(modelPath);
      
      String jsonString = await rootBundle.loadString(mapPath);
      mapGraph = jsonDecode(jsonString);
      
      isInitializing = false;
      scanStatus = "Starting scanner"; // Should flash briefly before the actual scan
    } catch (e) {
      // If a file is missing, it will turn the pill red and show this error!
      scanStatus = "Error: File missing";
      print("CRITICAL INIT ERROR: $e");
    }
  }

  Future<void> updateLocationAndRoute() async {
    if (isInitializing) return;

    final canScan = await WiFiScan.instance.canStartScan();
    
    switch (canScan) {
      case CanStartScan.yes:
        scanStatus = "Scanning";
        await WiFiScan.instance.startScan();
        final results = await WiFiScan.instance.getScannedResults();
        
        if (results.isEmpty) {
           scanStatus = "No Routers Found";
           return;
        }

        List<double> features = _formatWifiData(results);
        
        //predict path
        currentNodeId = aiEngine.predictLocation(features);
        // testing
        //currentNodeId = targetNodeId;
        
        scanStatus = "Node: $currentNodeId";
        calculateRoute();
        break;
        
      case CanStartScan.notSupported:
        scanStatus = "Error: WiFi Scan Not Supported";
        break;
      case CanStartScan.noLocationPermissionRequired:
      case CanStartScan.noLocationPermissionDenied:
      case CanStartScan.noLocationPermissionUpgradeAccuracy:
        scanStatus = "Error: Permission Missing";
        break;
      case CanStartScan.noLocationServiceDisabled:
        scanStatus = "Error: Turn on phone GPS";
        break;
      default:
        scanStatus = "Android Throttled, Waiting"; 
        break;
    }
  }

  void calculateRoute() {
    if (currentNodeId == -1 || targetNodeId == -1) return;
    // outside map use case
    if (!mapGraph.containsKey(currentNodeId.toString())) {
      scanStatus = "Warning: Off-Map ($currentNodeId)";
      nodesRemaining = 0;
      return; 
    }

    List<String> path = _findShortestPath(currentNodeId.toString(), targetNodeId.toString());
    
    if (path.isNotEmpty && path.length > 1) {
      nodesRemaining = path.length - 1; 
      String nextWaypoint = path[1];
      
      if (mapGraph.containsKey(currentNodeId.toString()) && mapGraph.containsKey(nextWaypoint)) {
        var currentLoc = mapGraph[currentNodeId.toString()];
        var nextLoc = mapGraph[nextWaypoint];
        
        currentFloor = currentLoc['floor'];
        nextFloor = nextLoc['floor'];

        targetBearing = _calculateBearing(
          (currentLoc['lat'] as num).toDouble(), (currentLoc['lon'] as num).toDouble(), 
          (nextLoc['lat'] as num).toDouble(), (nextLoc['lon'] as num).toDouble()
        );

        distanceToTarget = _calculatePathDistance(path);
      }
    } else if (path.length == 1 && currentNodeId == targetNodeId) {
      nodesRemaining = 0; 
      distanceToTarget = 0.0;
      nextFloor = currentFloor;
    }
  }

  List<double> _formatWifiData(List<WiFiAccessPoint> results) {
    return List.filled(128, -100.0); 
  }

  List<String> _findShortestPath(String start, String target) {
    if (start == target) return [start];
    if (!mapGraph.containsKey(start) || !mapGraph.containsKey(target)) return [];

    Queue<List<String>> queue = Queue();
    Set<String> visited = {};
    queue.add([start]);
    visited.add(start);

    while (queue.isNotEmpty) {
      List<String> path = queue.removeFirst();
      String current = path.last;
      
      if (current == target) return path;

      List<dynamic> neighbors = mapGraph[current]['neighbors'];
      for (String neighbor in neighbors) {
        if (!visited.contains(neighbor)) {
          visited.add(neighbor);
          queue.add(List.from(path)..add(neighbor));
        }
      }
    }
    return []; 
  }

  double _calculatePathDistance(List<String> path) {
    double totalDistance = 0.0;
    const double R = 6371000; 

    for (int i = 0; i < path.length - 1; i++) {
      var loc1 = mapGraph[path[i]];
      var loc2 = mapGraph[path[i+1]];
      
      double lat1 = (loc1['lat'] as num).toDouble() * math.pi / 180;
      double lat2 = (loc2['lat'] as num).toDouble() * math.pi / 180;
      double dLat = ((loc2['lat'] as num).toDouble() - (loc1['lat'] as num).toDouble()) * math.pi / 180;
      double dLon = ((loc2['lon'] as num).toDouble() - (loc1['lon'] as num).toDouble()) * math.pi / 180;

      double a = math.sin(dLat/2) * math.sin(dLat/2) +
                 math.cos(lat1) * math.cos(lat2) * math.sin(dLon/2) * math.sin(dLon/2);
      double c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a));
      totalDistance += R * c;
    }
    return totalDistance;
  }

  double _calculateBearing(double lat1, double lon1, double lat2, double lon2) {
    double dLon = (lon2 - lon1) * (math.pi / 180.0);
    lat1 = lat1 * (math.pi / 180.0);
    lat2 = lat2 * (math.pi / 180.0);
    double y = math.sin(dLon) * math.cos(lat2);
    double x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon);
    return (math.atan2(y, x) * (180.0 / math.pi) + 360) % 360;
  }
}