// src/MyMap.js
import React from "react";
import L from "leaflet";  // Import Leaflet to check layer types
import { MapContainer, TileLayer, FeatureGroup } from "react-leaflet";
import { EditControl } from "react-leaflet-draw";

function MyMap({ onCoordinatesSelected }) {
  const handleCreated = (e) => {
    const layer = e.layer;
    if (!layer) return;

    if (layer instanceof L.Rectangle) {
      // For a rectangle, use its bounding box
      const { _northEast, _southWest } = layer.getBounds();
      const north = _northEast.lat;
      const south = _southWest.lat;
      const east = _northEast.lng;
      const west = _southWest.lng;
      const coordsString = `${north},${south},${east},${west}`;
      onCoordinatesSelected(coordsString);
    } else if (layer instanceof L.Polygon) {
      // For a polygon, return all its vertices as comma-separated values
      const latLngs = layer.getLatLngs()[0]; // assuming single polygon
      const coordsString = latLngs
        .map((pt) => `${pt.lat},${pt.lng}`)
        .join(',');
      onCoordinatesSelected(coordsString);
    }
  };

  return (
    <MapContainer center={[41.4, 2.1]} zoom={12} style={{ width: "100%", height: "100%" }}>
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
      />
      <FeatureGroup>
        <EditControl
          position="topright"
          onCreated={handleCreated}
          draw={{
            polygon: true,
            rectangle: true,
            circle: false,
            circlemarker: false,
            marker: false,
            polyline: false,
          }}
        />
      </FeatureGroup>
    </MapContainer>
  );
}

export default MyMap;
