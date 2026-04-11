"""
CSI collector server (csi_collector_windows.py) running on the same local network, instead of scanning WiFi locally on Android.
"""
import 'dart:convert';
import 'dart:math' as math;
import 'dart:collection';
import 'package:flutter/services.dart';
import 'package:wifi_scan/wifi_scan.dart';
import 'package:http/http.dart' as http;
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

  String scanStatus = "Initializing...";
  bool isInitializing = true;

  String csiServerHost = "192.168.1.100"; // <-- CHANGE
  int    csiServerPort = 8765;
  bool   _serverReachable = false;
  bool   _lastDataStale   = false;

  String get serverUrl => "http://$csiServerHost:$csiServerPort";

  Future<void> initialize(String modelPath, String mapPath) async {
    try {
      isInitializing = true;
      scanStatus = "Loading assets...";
      await aiEngine.loadModel(modelPath);
      String jsonString = await rootBundle.loadString(mapPath);
      mapGraph = jsonDecode(jsonString);
      isInitializing = false;

      await _probeServer();

      scanStatus = _serverReachable
          ? "CSI Server: Connected"
          : "CSI Server: Not found — using RSSI";
    } catch (e) {
      scanStatus = "File missing";
      print("CRITICAL INIT ERROR: $e");
    }
  }

  Future<void> _probeServer() async {
    try {
      final response = await http
          .get(Uri.parse("$serverUrl/ping"))
          .timeout(const Duration(seconds: 2));
      _serverReachable = response.statusCode == 200;
    } catch (_) {
      _serverReachable = false;
    }
  }
  Future<void> updateLocationAndRoute() async {
    if (isInitializing) return;
    List<double>? features;
    if (_serverReachable) {
      features = await _fetchCSIFromServer();
    }
    if (features == null) {
      features = await _scanLocalRSSI();
    }
    if (features == null) {
      scanStatus = "No data source available";
      return;
    }
    currentNodeId = aiEngine.predictLocation(features);
    scanStatus = _serverReachable
        ? "CSI Node: $currentNodeId${_lastDataStale ? " [stale]" : ""}"
        : "RSSI Node: $currentNodeId";

    calculateRoute();
  }

  Future<List<double>?> _fetchCSIFromServer() async {
    try {
      final response = await http
          .get(Uri.parse("$serverUrl/csi"))
          .timeout(const Duration(seconds: 3));

      if (response.statusCode != 200) {
        _serverReachable = false;
        return null;
      }

      final Map<String, dynamic> body = jsonDecode(response.body);

      _lastDataStale = body["stale"] == true;

      final List<dynamic> raw = body["features"];
      if (raw.length != 128) {
        print("[CSI] Unexpected feature length: ${raw.length}");
        return null;
      }

      return raw.map<double>((v) => (v as num).toDouble()).toList();
    } catch (e) {
      print("[CSI] Server fetch failed: $e");
      _serverReachable = false;
      Future.delayed(const Duration(seconds: 5), _probeServer);
      return null;
    }
  }
  // fallback
  Future<List<double>?> _scanLocalRSSI() async {
    final canScan = await WiFiScan.instance.canStartScan();

    if (canScan != CanStartScan.yes) {
      switch (canScan) {
        case CanStartScan.notSupported:
          scanStatus = "Error: WiFi Scan Not Supported";
          break;
        case CanStartScan.noLocationPermissionRequired:
        case CanStartScan.noLocationPermissionDenied:
        case CanStartScan.noLocationPermissionUpgradeAccuracy:
          scanStatus = "Error: Permission Missing";
          break;
        case CanStartScan.noLocationServiceDisabled:
          scanStatus = "Error: Turn on phone GPS"; // needed since wifi scanning requires this, altho no GPS data is needed here technically
          break;
        default:
          scanStatus = "Android Throttled: Waiting...";
      }
      return null;
    }

    scanStatus = "Scanning WiFi (RSSI fallback)...";
    await WiFiScan.instance.startScan();
    final results = await WiFiScan.instance.getScannedResults();

    if (results.isEmpty) {
      scanStatus = "No Routers Found!";
      return null;
    }

    return _formatRSSIData(results);
  }

  
  List<double> _formatRSSIData(List<WiFiAccessPoint> results) {
    final Map<String, double> rssiMap = {};
    for (final ap in results) {
      rssiMap[ap.bssid.toUpperCase()] = ap.level.toDouble();
    }

    // TODO: Replace this placeholder list with the actual 128 BSSIDs
    // from selected_waps.json produced during training.
    // Until then, it returns a feature of all -105 (undetected sentinel).
    const List<String> selectedWAPs = [
      // "AA:BB:CC:DD:EE:01",
      // "AA:BB:CC:DD:EE:02",
      // ... add all 128 BSSIDs here
    ];

    if (selectedWAPs.isEmpty) {
      return List.filled(128, -105.0);
    }

    return selectedWAPs
        .map((bssid) => rssiMap[bssid] ?? -105.0)
        .toList();
  }

  void calculateRoute() {
    if (currentNodeId == -1 || targetNodeId == -1) return;

    if (!mapGraph.containsKey(currentNodeId.toString())) {
      scanStatus += " ⚠ Off-Map";
      nodesRemaining = 0;
      return;
    }

    List<String> path = _findShortestPath(
        currentNodeId.toString(), targetNodeId.toString());

    if (path.isNotEmpty && path.length > 1) {
      nodesRemaining = path.length - 1;
      String nextWaypoint = path[1];

      if (mapGraph.containsKey(currentNodeId.toString()) &&
          mapGraph.containsKey(nextWaypoint)) {
        var currentLoc = mapGraph[currentNodeId.toString()];
        var nextLoc    = mapGraph[nextWaypoint];

        currentFloor = currentLoc['floor'];
        nextFloor    = nextLoc['floor'];

        targetBearing = _calculateBearing(
          (currentLoc['lat'] as num).toDouble(),
          (currentLoc['lon'] as num).toDouble(),
          (nextLoc['lat'] as num).toDouble(),
          (nextLoc['lon'] as num).toDouble(),
        );

        distanceToTarget = _calculatePathDistance(path);
      }
    } else if (path.length == 1 && currentNodeId == targetNodeId) {
      nodesRemaining   = 0;
      distanceToTarget = 0.0;
      nextFloor        = currentFloor;
    }
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
    double total = 0.0;
    const double R = 6371000;
    for (int i = 0; i < path.length - 1; i++) {
      var loc1 = mapGraph[path[i]];
      var loc2 = mapGraph[path[i + 1]];
      double lat1 = (loc1['lat'] as num).toDouble() * math.pi / 180;
      double lat2 = (loc2['lat'] as num).toDouble() * math.pi / 180;
      double dLat = ((loc2['lat'] as num).toDouble() -
              (loc1['lat'] as num).toDouble()) *
          math.pi / 180;
      double dLon = ((loc2['lon'] as num).toDouble() -
              (loc1['lon'] as num).toDouble()) *
          math.pi / 180;
      double a = math.sin(dLat / 2) * math.sin(dLat / 2) +
          math.cos(lat1) *
              math.cos(lat2) *
              math.sin(dLon / 2) *
              math.sin(dLon / 2);
      total += R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a));
    }
    return total;
  }

  double _calculateBearing(
      double lat1, double lon1, double lat2, double lon2) {
    double dLon = (lon2 - lon1) * (math.pi / 180.0);
    lat1 = lat1 * (math.pi / 180.0);
    lat2 = lat2 * (math.pi / 180.0);
    double y = math.sin(dLon) * math.cos(lat2);
    double x = math.cos(lat1) * math.sin(lat2) -
        math.sin(lat1) * math.cos(lat2) * math.cos(dLon);
    return (math.atan2(y, x) * (180.0 / math.pi) + 360) % 360;
  }

  // ── Server config helper (can be called from UI settings screen) ──────────
  void setServerAddress(String host, {int port = 8765}) {
    csiServerHost    = host;
    csiServerPort    = port;
    _serverReachable = false;
    _probeServer();
  }
}