import 'dart:async';
import 'dart:math' as math;
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_compass/flutter_compass.dart';
import 'navigation_controller.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(const ThesisApp());
}

class Venue {
  final String name;
  final String modelPath;
  final String mapPath;
  final Map<String, int> destinations;
  Venue(this.name, this.modelPath, this.mapPath, this.destinations);
}

final List<Venue> supportedVenues = [
  Venue("University Building", "assets/wifi_model.tflite", "assets/map_graph.json", {"Computer Lab": 159, "Library": 305, "Cafeteria": 210}),
  Venue("Central Railway Station", "assets/railway_model.tflite", "assets/railway_map.json", {"Platform 1": 50, "Ticket Counter": 12, "Restrooms": 88}),
];

class ThesisApp extends StatelessWidget {
  const ThesisApp({Key? key}) : super(key: key);
  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: VenueSelectionScreen(),
    );
  }
}

class VenueSelectionScreen extends StatelessWidget {
  const VenueSelectionScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0F172A),
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Padding(
              padding: EdgeInsets.fromLTRB(20, 60, 20, 20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text("Indoor Navigator using wireless CSI", style: TextStyle(color: Colors.white, fontSize: 42, fontWeight: FontWeight.w900, height: 1.1, letterSpacing: 1.5)),
                  SizedBox(height: 10),
                  Text("By Sai Sleghana Bala", style: TextStyle(color: Colors.greenAccent, fontSize: 16, fontWeight: FontWeight.w600)),
                  SizedBox(height: 30),
                  Text("SELECT YOUR LOCATION", style: TextStyle(color: Colors.white54, fontSize: 12, letterSpacing: 2, fontWeight: FontWeight.bold)),
                ],
              ),
            ),
            Expanded(
              child: ListView.builder(
                itemCount: supportedVenues.length,
                itemBuilder: (context, index) {
                  final venue = supportedVenues[index];
                  return Container(
                    margin: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
                    decoration: BoxDecoration(color: Colors.white.withOpacity(0.05), borderRadius: BorderRadius.circular(20), border: Border.all(color: Colors.white.withOpacity(0.1))),
                    child: ListTile(
                      contentPadding: const EdgeInsets.all(15),
                      leading: Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(color: Colors.blueAccent.withOpacity(0.2), shape: BoxShape.circle),
                        child: const Icon(Icons.business, size: 28, color: Colors.blueAccent),
                      ),
                      title: Text(venue.name, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white)),
                      subtitle: Text("${venue.destinations.length} Supported Zones", style: const TextStyle(color: Colors.white54)),
                      trailing: const Icon(Icons.arrow_forward_ios, color: Colors.white38),
                      onTap: () {
                        Navigator.push(context, MaterialPageRoute(builder: (context) => ARScreen(activeVenue: venue)));
                      },
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class ARScreen extends StatefulWidget {
  final Venue activeVenue;
  const ARScreen({Key? key, required this.activeVenue}) : super(key: key);
  @override
  _ARScreenState createState() => _ARScreenState();
}

class _ARScreenState extends State<ARScreen> {
  late CameraController _cameraController;
  final NavigationController _navController = NavigationController();
  
  double _compassHeading = 0.0;
  Timer? _locationTimer;
  late String _selectedDestinationName;

  @override
  void initState() {
    super.initState();
    _selectedDestinationName = widget.activeVenue.destinations.keys.first;
    _navController.targetNodeId = widget.activeVenue.destinations[_selectedDestinationName]!;

    _cameraController = CameraController(cameras[0], ResolutionPreset.high);
    _cameraController.initialize().then((_) {
      if (!mounted) return;
      setState(() {});
    });

    FlutterCompass.events?.listen((event) {
      if (mounted) setState(() => _compassHeading = event.heading ?? 0.0);
    });

    _navController.initialize(widget.activeVenue.modelPath, widget.activeVenue.mapPath).then((_) {
      _locationTimer = Timer.periodic(const Duration(seconds: 2), (timer) async {
        await _navController.updateLocationAndRoute();
        setState(() {}); 
      });
    });
  }

  @override
  void dispose() {
    _cameraController.dispose();
    _locationTimer?.cancel();
    super.dispose();
  }

  String _getTurnInstruction(double targetBearing, double currentHeading) {
    double diff = (targetBearing - currentHeading);
    while (diff <= -180) diff += 360;
    while (diff > 180) diff -= 360;

    if (diff > -25 && diff < 25) return "Go Straight";
    if (diff >= 25 && diff <= 150) return "Turn Right";
    if (diff <= -25 && diff >= -150) return "Turn Left";
    return "Turn Around";
  }

  @override
  Widget build(BuildContext context) {
    if (!_cameraController.value.isInitialized) {
      return const Scaffold(backgroundColor: Colors.black, body: Center(child: CircularProgressIndicator(color: Colors.greenAccent)));
    }

    bool hasArrived = _navController.currentNodeId == _navController.targetNodeId;
    double arrowRotationRadians = (_navController.targetBearing - _compassHeading) * (math.pi / 180);
    bool needsFloorChange = _navController.currentFloor != _navController.nextFloor && !hasArrived;
    
    String turnInstruction = hasArrived ? "You have arrived!" : _getTurnInstruction(_navController.targetBearing, _compassHeading);

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          CameraPreview(_cameraController),

          Positioned(
            top: 60, left: 20, right: 20,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(20),
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 5),
                  decoration: BoxDecoration(color: Colors.black.withOpacity(0.5), borderRadius: BorderRadius.circular(20), border: Border.all(color: Colors.white.withOpacity(0.2))),
                  child: DropdownButtonHideUnderline(
                    child: DropdownButton<String>(
                      isExpanded: true,
                      dropdownColor: Colors.grey[900],
                      value: _selectedDestinationName,
                      icon: const Icon(Icons.flag, color: Colors.greenAccent),
                      style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
                      items: widget.activeVenue.destinations.keys.map((String locName) {
                        return DropdownMenuItem<String>(value: locName, child: Text(locName));
                      }).toList(),
                      onChanged: (String? newValue) {
                        if (newValue != null) {
                          setState(() {
                            _selectedDestinationName = newValue;
                            _navController.targetNodeId = widget.activeVenue.destinations[newValue]!;
                            _navController.calculateRoute();
                          });
                        }
                      },
                    ),
                  ),
                ),
              ),
            ),
          ),

          Center(
            child: hasArrived
                ? const Icon(Icons.check_circle, size: 200, color: Colors.greenAccent, shadows: [Shadow(color: Colors.black, blurRadius: 20)])
                : Transform.rotate(
                    angle: arrowRotationRadians,
                    child: Icon(Icons.arrow_upward_rounded, size: 250, color: needsFloorChange ? Colors.redAccent : Colors.white, shadows: const [Shadow(color: Colors.black, blurRadius: 20)]),
                  ),
          ),

          if (needsFloorChange)
            Positioned(
              top: 150, left: 20, right: 20,
              child: Container(
                padding: const EdgeInsets.all(15),
                decoration: BoxDecoration(color: Colors.redAccent, borderRadius: BorderRadius.circular(10), boxShadow: const [BoxShadow(color: Colors.black45, blurRadius: 10)]),
                child: Text(
                  _navController.nextFloor > _navController.currentFloor ? "🔼 Take Stairs UP to Floor ${_navController.nextFloor}" : "🔽 Take Stairs DOWN to Floor ${_navController.nextFloor}",
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold),
                ),
              ),
            ),

          Positioned(
            bottom: 40, left: 20, right: 20,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(20),
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                child: Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(color: Colors.black.withOpacity(0.6), borderRadius: BorderRadius.circular(20), border: Border.all(color: Colors.white.withOpacity(0.2))),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        turnInstruction,
                        style: TextStyle(color: hasArrived ? Colors.greenAccent : Colors.white, fontSize: 32, fontWeight: FontWeight.w900),
                      ),
                      const SizedBox(height: 10),
                      
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Row(
                            children: [
                              Icon(
                                hasArrived ? Icons.flag : Icons.route, 
                                color: hasArrived ? Colors.greenAccent : Colors.blueAccent, 
                                size: 22
                              ),
                              const SizedBox(width: 8),
                              Text(
                                hasArrived ? "Destination Reached" : "${_navController.nodesRemaining} Waypoints Left",
                                style: TextStyle(
                                  color: hasArrived ? Colors.greenAccent : Colors.blueAccent, 
                                  fontSize: 16, 
                                  fontWeight: FontWeight.bold
                                ),
                              ),
                            ],
                          ),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                            decoration: BoxDecoration(
                              color: _navController.scanStatus.contains("Error") || _navController.scanStatus.contains("Throttled")
                                  ? Colors.redAccent.withOpacity(0.2) 
                                  : Colors.greenAccent.withOpacity(0.2),
                              borderRadius: BorderRadius.circular(20),
                              border: Border.all(
                                color: _navController.scanStatus.contains("Error") || _navController.scanStatus.contains("Throttled")
                                  ? Colors.redAccent 
                                  : Colors.greenAccent,
                                width: 1.5,
                              ),
                            ),
                            child: Text(
                              _navController.scanStatus,
                              style: TextStyle(
                                color: _navController.scanStatus.contains("Error") || _navController.scanStatus.contains("Throttled")
                                    ? Colors.redAccent 
                                    : Colors.greenAccent, 
                                fontSize: 12,
                                fontWeight: FontWeight.w900,
                                letterSpacing: 0.5,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}