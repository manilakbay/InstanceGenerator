import React, { useState, useMemo, useRef } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";
import { Bar } from "react-chartjs-2";
import Chart from "chart.js/auto";
import MyMap from "./MyMap";
import Accordion from "react-bootstrap/Accordion";
import Modal from "react-bootstrap/Modal";
import Button from "react-bootstrap/Button";
import Joyride, { STATUS, EVENTS } from "react-joyride";

function App() {
  const [formData, setFormData] = useState({
    // General Settings
    areaSelectionMethod: "address",
    area: "",
    coordinates: "",
    instanceName: "",
    show_arrows: false,
    numDepots: 1,
    depotChoice: "default",
    numCustomers: 10,
    customerChoice: "supermarket",
    numChargingStations: 5,
    chargingStationChoice: "default",
    // simplify: false,

    //Demand & Service Time
    demandOption: "default",    // "default" | "constant" | "random"
    demandConstant: 100,
    demandRangeMin: 50,
    demandRangeMax: 240,
    serviceTimeOption: "default",   // "default" | "constant" | "random"
    serviceTimeConstant: 0.5,
    serviceTimeMin: 0.1,
    serviceTimeMax: 1.0,

    //Vehicle Config
    batteryCapacityOption: "default",    // "default" | "user"
    batteryCapacityValue: 100.0,
    loadCapacityOption: "default",       // "default" | "user"
    loadCapacityValue: 1000.0,
    charginRateOption: "default",  // "default" | "user"
    chargingRateValue: 0.2,

    // Time Limit Field (In Hours): 
    tourTimeLimit: 8,

    // Seed for the random number generator
    randomSeed: 42
  });

  const [responseMessage, setResponseMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [resultLinks, setResultLinks] = useState(null);
  const [summaryData, setSummaryData] = useState(null);
  const [generatedKey, setGeneratedKey] = useState(Date.now());
  const [showingResultMap, setShowingResultMap] = useState(false);
  const [showResultsDrawer, setShowResultsDrawer] = useState(false);
  const [isDrawerCollapsed, setIsDrawerCollapsed] = useState(false);
  const [showInfoModal, setShowInfoModal] = useState(false);
  const [runTour, setRunTour] = useState(false);
  
  // Generation mode state: 'random' or 'upload'
  const [generationMode, setGenerationMode] = useState('random');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [fileStats, setFileStats] = useState({ depots: 0, customers: 0 });
  const fileInputRef = useRef(null);

  // API Base URL - uses environment variable or same origin (works in both dev and production)
  const API_BASE_URL = process.env.REACT_APP_API_URL || window.location.origin;

  // Helper function to format error messages for display
  const formatErrorMessage = (errorMsg) => {
    if (!errorMsg) return errorMsg;
    
    // Replace line breaks with HTML breaks for better formatting
    let formatted = errorMsg.replace(/\n/g, '<br>');
    
    // Style the warning emoji
    formatted = formatted.replace(/⚠️/g, '<span style="font-size: 1.2em;">⚠️</span>');
    
    // Make headings bold
    formatted = formatted.replace(/(INSUFFICIENT[^:]+:)/g, '<strong>$1</strong>');
    formatted = formatted.replace(/(SOLUTIONS?:)/g, '<strong>$1</strong>');
    formatted = formatted.replace(/(ASSIGNMENT ERROR:)/g, '<strong>$1</strong>');
    formatted = formatted.replace(/(COORDINATE VALIDATION ERROR:)/g, '<strong>$1</strong>');
    
    // Style numbered lists (1., 2., 3., etc.)
    formatted = formatted.replace(/(\d+\.\s)/g, '<strong>$1</strong>');
    
    return formatted;
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === "checkbox" ? checked : value,
    });
  };

  // Callback from MyMap component
  const handleCoordinatesSelected = (coordsString) => {
    setFormData((prev) => ({
      ...prev,
      areaSelectionMethod: "coordinates",
      coordinates: coordsString,
    }));
  };

  // Handle file upload and parsing
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploadedFile(file);

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target.result;
      try {
        const lines = text.split('\n').filter(line => line.trim() !== '');
        
        // Check if first line is header
        const headerLine = lines[0].toLowerCase();
        let dataStartIndex = 0;
        if (headerLine.includes('id') && headerLine.includes('type') && 
            (headerLine.includes('x') || headerLine.includes('latitude'))) {
          dataStartIndex = 1; // Skip header
        }

        // Parse CSV
        let depots = 0;
        let customers = 0;
        
        for (let i = dataStartIndex; i < lines.length; i++) {
          const line = lines[i].trim();
          if (!line) continue;
          
          // Handle both comma and tab separators
          const parts = line.split(/[,\t]/).map(p => p.trim());
          
          // Look for type column (could be index 1 or by header)
          // Format: id,type,x,y,demand,service_time
          // Or: id,type,latitude,longitude,demand,service_time
          if (parts.length >= 3) {
            const typeValue = parts[1].toLowerCase();
            if (typeValue === 'd' || typeValue === 'depot') {
              depots++;
            } else if (typeValue === 'c' || typeValue === 'customer') {
              customers++;
            }
          }
        }

        setFileStats({ depots, customers });
        
        // Auto-update form data with counts
        setFormData((prev) => ({
          ...prev,
          numDepots: depots > 0 ? depots : prev.numDepots,
          numCustomers: customers > 0 ? customers : prev.numCustomers,
        }));
      } catch (error) {
        console.error('Error parsing file:', error);
        setFileStats({ depots: 0, customers: 0 });
      }
    };
    reader.readAsText(file);
  };

  // Download template CSV file
  const downloadTemplate = () => {
    const templateContent = `id,type,x,y,demand,service_time
0,d,2.1011217,41.5661504,0,0.0
1,c,2.0816034,41.5565825,120,0.5
2,c,2.1091703,41.5462223,90,0.5`;

    const blob = new Blob([templateContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'coordinate_template.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  // Clear uploaded file
  const clearUploadedFile = () => {
    setUploadedFile(null);
    setFileStats({ depots: 0, customers: 0 });
    // Reset file input element
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    // Reset to default values
    setFormData((prev) => ({
      ...prev,
      numDepots: 1,
      numCustomers: 10,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResponseMessage("");
    setResultLinks(null);
    setSummaryData(null);
    setShowingResultMap(false);
    setShowResultsDrawer(false);
    setIsDrawerCollapsed(false);

    try {
      // Prepare request data
      const requestData = { ...formData, generationMode };
      
      // If upload mode, include file data
      if (generationMode === 'upload' && uploadedFile) {
        const reader = new FileReader();
        reader.onload = async (event) => {
          const fileContent = event.target.result;
          requestData.uploadedFileContent = fileContent;
          requestData.uploadedFileName = uploadedFile.name;
          
          // Send request with file content
          await sendGenerationRequest(requestData);
        };
        reader.readAsText(uploadedFile);
        return; // Exit early, request will be sent from reader.onload
      } else {
        // Normal random generation
        await sendGenerationRequest(requestData);
      }
    } catch (error) {
      console.error("Error submitting form:", error);
      setResponseMessage("An error occurred while processing your request.");
      setLoading(false);
    }
  };

  // Separate function to send the actual request
  const sendGenerationRequest = async (requestData) => {
    try {
      const response = await fetch(`${API_BASE_URL}/generate-instance`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestData),
      });

      const data = await response.json();

      if (response.ok) {
        setResultLinks({
          dataset: `${API_BASE_URL}${data.dataset_url}`,
          htmlVisualization: `${API_BASE_URL}${data.html_visualization_url}`,
          road_colors: data.road_colors,
        });
        setSummaryData(data.summary_data);
        setResponseMessage(data.message);
        setShowingResultMap(true); // Automatically switch to result map view
        setShowResultsDrawer(true); // Show results drawer
        setIsDrawerCollapsed(false); // Open drawer in expanded state
        console.log("Full Response Data:", data);
      } else {
        // Check if it's a structured error
        if (data.error && (data.type === 'validation_error' || data.type === 'generation_error')) {
          // Format validation errors nicely
          setResponseMessage(data.message);
        } else {
          setResponseMessage(
            data.message || "Error occurred while generating the instance."
          );
        }
        // Show results drawer even on error so user can see the error message
        setShowResultsDrawer(true);
        setIsDrawerCollapsed(false);
      }
    } catch (error) {
      console.error("Error submitting form:", error);
      setResponseMessage("An error occurred while processing your request.");
    } finally {
      setLoading(false);
    }
  };

  // Prepare general statistics
  const generalStats = summaryData
    ? [
        {
          label: "Total number of depots",
          value: summaryData["Total number of depots"],
        },
        {
          label: "Total number of customers",
          value: summaryData["Total number of customers"],
        },
        {
          label: "Total number of charging stations",
          value: summaryData["Total number of charging stations"],
        },
        {
          label: "Total number of road crossings",
          value: summaryData["Total number of road crossings"],
        },
        {
          label: "Total number of edges (road segments)",
          value: summaryData["Total number of edges (road segments)"],
        },
      ]
    : [];

  // Build chart data
  const roadTypeChartData =
    summaryData &&
    resultLinks?.road_colors &&
    Object.keys(summaryData).length > 0
      ? {
          labels: Object.keys(summaryData).filter((key) =>
            key.match(/motorway|link|primary|secondary|tertiary/i)
          ),
          datasets: [
            {
              label: "Number of Road Segments",
              backgroundColor: Object.keys(summaryData)
                .filter((key) =>
                  key.match(/motorway|link|primary|secondary|tertiary/i)
                )
                .map((key) => resultLinks.road_colors[key] || "#007bff"),
              data: Object.keys(summaryData)
                .filter((key) =>
                  key.match(/motorway|link|primary|secondary|tertiary/i)
                )
                .map((key) => parseInt(summaryData[key], 10) || 0),
            },
          ],
        }
      : null;

  // Handler for tab selection: if "Generated Instance" tab is selected, update the key to force re-mount
  const handleTabSelect = (key) => {
    if (key === "generated") {
      setGeneratedKey(Date.now());
    }
  };

  // Guided Tour Steps
  const tourSteps = useMemo(() => [
    {
      target: 'body',
      placement: 'center',
      title: 'Welcome to EVRPGen! 🚀',
      content: (
        <div>
          <p>This tool allows you to generate and visualize realistic instances of the <strong>Electric Vehicle Routing Problem with Road Junctions and Road Types</strong>.</p>
          <p className="mb-0">Let's take a quick tour to get you started!</p>
        </div>
      ),
      disableBeacon: true,
    },
    {
      target: '.sidebar-content',
      placement: 'right',
      title: 'Configuration Panel',
      content: (
        <div>
          <p>Configure your problem instance here using the accordion sections:</p>
          <ul className="text-start mb-0" style={{ fontSize: '0.9rem' }}>
            <li><strong>📍 General Settings:</strong> Select area by address or coordinates</li>
            <li><strong>📦 Node Settings:</strong> Set depots, customers, and charging stations</li>
            <li><strong>📊 Customer Demand:</strong> Configure demand values</li>
            <li><strong>⏱️ Service Time:</strong> Set service time parameters</li>
            <li><strong>🚗 Vehicle Config:</strong> Customize battery, load, and charging rates</li>
            <li><strong>⏰ Tour Time Limit:</strong> Set maximum tour duration</li>
          </ul>
        </div>
      ),
    },
    {
      target: '.map-container',
      placement: 'left',
      title: 'Interactive Map',
      content: (
        <div>
          <p>This is the interactive map area where you can:</p>
          <ul className="text-start mb-0" style={{ fontSize: '0.9rem' }}>
            <li>Draw bounding boxes or polygons to define your area (if using coordinates)</li>
            <li>Visualize the generated instance with depots, customers, and charging stations</li>
            <li>View the road network with different road types color-coded</li>
          </ul>
        </div>
      ),
    },
    {
      target: '.sidebar-footer',
      placement: 'top',
      title: 'Generate Instance',
      content: (
        <div>
          <p>Once you've configured all your settings, click this button to generate your instance.</p>
          <p className="mb-0">The generation process will extract the road network from OpenStreetMap and create a problem instance based on your parameters.</p>
        </div>
      ),
    },
    {
      target: '.map-toggle-container',
      placement: 'bottom',
      title: 'Map View Toggle',
      content: (
        <div>
          <p>After generating an instance, these buttons allow you to switch between:</p>
          <ul className="text-start mb-0" style={{ fontSize: '0.9rem' }}>
            <li><strong>Input Map:</strong> The original map for area selection</li>
            <li><strong>Result Map:</strong> The generated instance visualization</li>
          </ul>
        </div>
      ),
      disableBeacon: true,
    },
  ], []);

  // Joyride callback handler
  const handleJoyrideCallback = (data) => {
    const { status, type } = data;
    
    if (status === STATUS.FINISHED || status === STATUS.SKIPPED) {
      setRunTour(false);
    }
    
    // Log tour events for debugging
    if (type === EVENTS.STEP_AFTER || type === EVENTS.TARGET_NOT_FOUND) {
      // Handle step navigation or missing targets
      if (type === EVENTS.TARGET_NOT_FOUND && data.step?.target === '.map-toggle-container') {
        // Skip map toggle step if it doesn't exist yet (only appears after generation)
        // This is handled automatically by Joyride
      }
    }
  };

  return (
    <div className="app-container">
      {/* LEFT SIDEBAR */}
      <div className="sidebar d-flex flex-column">
        {/* Sidebar Header */}
        <div className="sidebar-header">
          <div className="d-flex justify-content-between align-items-center">
            <img
              src="/logos/ApplicationLogo.png"
              alt="EVRPGen Logo"
              style={{ maxHeight: '80px', width: 'auto' }}
              className="sidebar-logo"
            />
            <Button
              variant="link"
              className="info-button"
              onClick={() => setShowInfoModal(true)}
              aria-label="Show application information"
              title="About EVRPGen"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="16" x2="12" y2="12"></line>
                <line x1="12" y1="8" x2="12.01" y2="8"></line>
              </svg>
            </Button>
          </div>
        </div>

        {/* Scrollable Form Content */}
        <div className="sidebar-content">
          <Accordion defaultActiveKey="0" flush>
            <Accordion.Item eventKey="0">
              <Accordion.Header>
                <span className="accordion-icon">📍</span>
                General Settings
              </Accordion.Header>
                  <Accordion.Body>
                    {/* Generation Mode Toggle */}
                    <div className="mb-4">
                      <label className="form-label fw-bold">Generation Mode:</label>
                      <div className="generation-mode-toggle">
                        <button
                          type="button"
                          className={`mode-toggle-btn ${generationMode === 'random' ? 'active' : ''}`}
                          onClick={() => {
                            setGenerationMode('random');
                            clearUploadedFile();
                          }}
                        >
                          🎲 Generate Random
                        </button>
                        <button
                          type="button"
                          className={`mode-toggle-btn ${generationMode === 'upload' ? 'active' : ''}`}
                          onClick={() => setGenerationMode('upload')}
                        >
                          📂 Upload File
                        </button>
                      </div>
                    </div>

                    {/* File Upload Section - Only shown in upload mode */}
                    {generationMode === 'upload' && (
                      <div className="mb-4 p-3 border rounded bg-light">
                        <label className="form-label fw-bold">Upload Coordinates File:</label>
                        <div className="mb-2">
                          <input
                            type="file"
                            accept=".csv,.txt"
                            className="form-control"
                            onChange={handleFileUpload}
                            id="coordinateFileInput"
                            ref={fileInputRef}
                          />
                        </div>
                        {uploadedFile && (
                          <div className="mb-2">
                            <div className="d-flex justify-content-between align-items-center">
                              <small className="text-success">
                                ✓ File uploaded: <strong>{uploadedFile.name}</strong>
                              </small>
                              <button
                                type="button"
                                className="btn btn-sm btn-outline-danger"
                                onClick={clearUploadedFile}
                              >
                                Clear
                              </button>
                            </div>
                            {(fileStats.depots > 0 || fileStats.customers > 0) && (
                              <div className="mt-2">
                                <small className="text-muted">
                                  Found: <strong>{fileStats.depots}</strong> depot(s), <strong>{fileStats.customers}</strong> customer(s)
                                </small>
                              </div>
                            )}
                          </div>
                        )}
                        <div className="mt-2">
                          <button
                            type="button"
                            className="btn btn-sm btn-outline-primary"
                            onClick={downloadTemplate}
                          >
                            📥 Download Template
                          </button>
                          <small className="text-muted d-block mt-1">
                            Format: id,type,x,y,demand,service_time (type: 'd' for depot, 'c' for customer)
                          </small>
                        </div>
                      </div>
                    )}

                    {/* Area Selection - Always visible, but with note in upload mode */}
                    <div className="mb-3">
                      <label className="form-label">
                        {generationMode === 'upload' ? 'Map Region (for context):' : 'Select Area Method:'}
                      </label>
                      {generationMode === 'upload' && (
                        <small className="text-muted d-block mb-2">
                          Define the map region to extract the road network. Your uploaded coordinates should be within this area.
                        </small>
                      )}
                      <div>
                        <div className="form-check">
                          <input
                            type="radio"
                            name="areaSelectionMethod"
                            value="address"
                            className="form-check-input"
                            checked={formData.areaSelectionMethod === "address"}
                            onChange={handleChange}
                          />
                          <label className="form-check-label">By Address</label>
                        </div>
                        <div className="form-check">
                          <input
                            type="radio"
                            name="areaSelectionMethod"
                            value="coordinates"
                            className="form-check-input"
                            checked={formData.areaSelectionMethod === "coordinates"}
                            onChange={handleChange}
                          />
                          <label className="form-check-label">By Coordinates</label>
                        </div>
                      </div>
                    </div>
                    {formData.areaSelectionMethod === "address" && (
                      <div className="mb-3">
                        <label className="form-label">Area Name:</label>
                        <input
                          type="text"
                          name="area"
                          className="form-control"
                          value={formData.area}
                          onChange={handleChange}
                          placeholder="e.g., Barcelona, Spain"
                          required
                        />
                      </div>
                    )}
                    {formData.areaSelectionMethod === "coordinates" && (
                      <div className="mb-3">
                        <label className="form-label">
                          Coordinates (Enter comma‐separated values: For bounding box use 4 numbers (N,S,E,W); for polygon use 6 or more)
                        </label>
                        <textarea
                          name="coordinates"
                          className="form-control"
                          value={formData.coordinates}
                          onChange={handleChange}
                          placeholder="e.g., 41.5,41.3,2.2,2.0 (bounding box) or 41.6,41.5,2.1,2.0,2.2,2.1 (polygon)"
                          rows="3"
                          required
                        />
                      </div>
                    )}

                    {/* New Instance Name Field */}
                    <div className="mb-3">
                      <label className="form-label">Instance Name:</label>
                      <input
                        type="text"
                        name="instanceName"
                        className="form-control"
                        value={formData.instanceName}
                        onChange={handleChange}
                        placeholder="Enter a name for your instance"
                      />
                    </div>

                    <div className="form-check mb-3">
                    <input
                      type="checkbox"
                      name="show_arrows"
                      className="form-check-input"
                      checked={formData.show_arrows}
                      onChange={handleChange}
                    />
                    <label className="form-check-label">Show Arrows on Map</label>
                    </div>
                      <div className="mb-3">
                      <label className="form-label">Random Seed:</label>
                      <input
                        type="number"
                        name="randomSeed"
                        className="form-control"
                        value={formData.randomSeed}
                        onChange={handleChange}
                        min="0"
                        required
                      />
                      <small className="text-muted">
                        Use a specific integer seed for reproducibility. Default is 42.
                      </small>
                    </div>
                  </Accordion.Body>
                </Accordion.Item>

                {/* Node Settings - Hidden/Disabled in upload mode */}
                {generationMode === 'random' && (
                <Accordion.Item eventKey="2">
                  <Accordion.Header>
                    <span className="accordion-icon">📦</span>
                    Node Settings
                  </Accordion.Header>
                  <Accordion.Body>
                  <div className="mb-3">
                      <label className="form-label">Number of Depots:</label>
                      <input
                        type="number"
                        name="numDepots"
                        className="form-control"
                        value={formData.numDepots}
                        onChange={handleChange}
                        min="1"
                        required
                      />
                    </div>
                    {/* Added Depot Choice Dropdown */}
                    <div className="mb-3">
                      <label className="form-label">Depot Choice:</label>
                      <select
                        name="depotChoice"
                        className="form-select"
                        value={formData.depotChoice}
                        onChange={handleChange}
                      >
                        <option value="default">
                          Default (Warehouses, Industrial Facilities)
                        </option>
                      </select>
                    </div>
                    <div className="mb-3">
                      <label className="form-label">Number of Customers:</label>
                      <input
                        type="number"
                        name="numCustomers"
                        className="form-control"
                        value={formData.numCustomers}
                        onChange={handleChange}
                        min="1"
                        required
                      />
                    </div>
                    {/* Added Customer Choice Dropdown */}
                    <div className="mb-3">
                      <label className="form-label">Customer Choice:</label>
                      <select
                        name="customerChoice"
                        className="form-select"
                        value={formData.customerChoice}
                        onChange={handleChange}
                      >
                        <option value="supermarket">Supermarket</option>
                        <option value="bakery">Bakery</option>
                        <option value="convenience">Convenience Store</option>
                        <option value="pharmacy">Pharmacy</option>
                        <option value="hospital">Hospital</option>
                        <option value="school">School</option>
                        <option value="cafe">Cafe</option>
                        <option value="hotel">Hotel</option>
                      </select>
                    </div>
                    <div className="mb-3">
                      <label className="form-label">Number of Charging Stations:</label>
                      <input
                        type="number"
                        name="numChargingStations"
                        className="form-control"
                        value={formData.numChargingStations}
                        onChange={handleChange}
                        min="1"
                        required
                      />
                    </div>
                    {/* Added Charging Station Choice Dropdown */}
                    <div className="mb-3">
                      <label className="form-label">Charging Station Choice:</label>
                      <select
                        name="chargingStationChoice"
                        className="form-select"
                        value={formData.chargingStationChoice}
                        onChange={handleChange}
                      >
                        <option value="default">
                          Default (Gas Stations, Charging Stations)
                        </option>
                      </select>
                    </div>
                    {/* <div className="form-check mb-3">
                      <input
                        type="checkbox"
                        name="simplify"
                        className="form-check-input"
                        checked={formData.simplify}
                        onChange={handleChange}
                      />
                      <label className="form-check-label">Simplify Network</label>
                    </div> */}

                  </Accordion.Body>
                </Accordion.Item>
                )}

                {/* Node Settings - Read-only in upload mode */}
                {generationMode === 'upload' && (
                <Accordion.Item eventKey="2">
                  <Accordion.Header>
                    <span className="accordion-icon">📦</span>
                    Node Settings (from file)
                  </Accordion.Header>
                  <Accordion.Body>
                    <div className="mb-3">
                      <label className="form-label">Number of Depots:</label>
                      <input
                        type="number"
                        name="numDepots"
                        className="form-control"
                        value={formData.numDepots}
                        onChange={handleChange}
                        min="1"
                        disabled
                        style={{ backgroundColor: '#e9ecef', cursor: 'not-allowed' }}
                      />
                      <small className="text-muted">Set automatically from uploaded file</small>
                    </div>
                    <div className="mb-3">
                      <label className="form-label">Number of Customers:</label>
                      <input
                        type="number"
                        name="numCustomers"
                        className="form-control"
                        value={formData.numCustomers}
                        onChange={handleChange}
                        min="1"
                        disabled
                        style={{ backgroundColor: '#e9ecef', cursor: 'not-allowed' }}
                      />
                      <small className="text-muted">Set automatically from uploaded file</small>
                    </div>
                    <div className="mb-3">
                      <label className="form-label">Number of Charging Stations:</label>
                      <input
                        type="number"
                        name="numChargingStations"
                        className="form-control"
                        value={formData.numChargingStations}
                        onChange={handleChange}
                        min="1"
                        required
                      />
                      <small className="text-muted">Charging stations will be generated randomly</small>
                    </div>
                    {/* Added Charging Station Choice Dropdown */}
                    <div className="mb-3">
                      <label className="form-label">Charging Station Choice:</label>
                      <select
                        name="chargingStationChoice"
                        className="form-select"
                        value={formData.chargingStationChoice}
                        onChange={handleChange}
                      >
                        <option value="default">
                          Default (Gas Stations, Charging Stations)
                        </option>
                      </select>
                    </div>
                  </Accordion.Body>
                </Accordion.Item>
                )}

                {/* Customer Demand - Hidden in upload mode */}
                {generationMode === 'random' && (
                <Accordion.Item eventKey="3">
                  <Accordion.Header>
                    <span className="accordion-icon">📊</span>
                    Customer Demand
                  </Accordion.Header>
                  <Accordion.Body>
                    <div className="mb-3">
                      <label className="form-label">Demand Option:</label>
                      <select
                        className="form-select"
                        name="demandOption"
                        value={formData.demandOption}
                        onChange={handleChange}
                      >
                        <option value="default">
                          Default (random from 50..240 in steps of 10)
                        </option>
                        <option value="constant">Constant</option>
                        <option value="random">Random</option>
                      </select>
                    </div>
                    {formData.demandOption === "constant" && (
                      <div className="mb-3">
                        <label className="form-label">Constant Demand Value:</label>
                        <input
                          type="number"
                          className="form-control"
                          name="demandConstant"
                          value={formData.demandConstant}
                          onChange={handleChange}
                          min="0"
                        />
                      </div>
                    )}
                    {formData.demandOption === "random" && (
                      <div className="mb-3">
                        <label className="form-label">Demand Range Min:</label>
                        <input
                          type="number"
                          className="form-control"
                          name="demandRangeMin"
                          value={formData.demandRangeMin}
                          onChange={handleChange}
                        />
                        <label className="mt-2">Demand Range Max:</label>
                        <input
                          type="number"
                          className="form-control"
                          name="demandRangeMax"
                          value={formData.demandRangeMax}
                          onChange={handleChange}
                        />
                      </div>
                    )}
                  </Accordion.Body>
                </Accordion.Item>
                )}

                {/* Service Time - Hidden in upload mode */}
                {generationMode === 'random' && (
                <Accordion.Item eventKey="4">
                  <Accordion.Header>
                    <span className="accordion-icon">⏱️</span>
                    Service Time
                  </Accordion.Header>
                  <Accordion.Body>
                    <div className="mb-3">
                      <label className="form-label">Service Time Option:</label>
                      <select
                        className="form-select"
                        name="serviceTimeOption"
                        value={formData.serviceTimeOption}
                        onChange={handleChange}
                      >
                        <option value="default">
                          Default (0.5 hours per customer)
                        </option>
                        <option value="constant">Constant</option>
                        <option value="random">Random</option>
                      </select>
                    </div>
                    {formData.serviceTimeOption === "constant" && (
                      <div className="mb-3">
                        <label className="form-label">
                          Constant Service Time (hours):
                        </label>
                        <input
                          type="number"
                          step="0.01"
                          className="form-control"
                          name="serviceTimeConstant"
                          value={formData.serviceTimeConstant}
                          onChange={handleChange}
                        />
                      </div>
                    )}
                    {formData.serviceTimeOption === "random" && (
                      <div className="mb-3">
                        <label className="form-label">
                          Service Time Min (hours):
                        </label>
                        <input
                          type="number"
                          step="0.01"
                          className="form-control"
                          name="serviceTimeMin"
                          value={formData.serviceTimeMin}
                          onChange={handleChange}
                        />
                        <label className="mt-2">
                          Service Time Max (hours):
                        </label>
                        <input
                          type="number"
                          step="0.01"
                          className="form-control"
                          name="serviceTimeMax"
                          value={formData.serviceTimeMax}
                          onChange={handleChange}
                        />
                      </div>
                    )}
                  </Accordion.Body>
                </Accordion.Item>
                )}
                <Accordion.Item eventKey="5">
                  <Accordion.Header>
                    <span className="accordion-icon">🚗</span>
                    Vehicle Configuration
                  </Accordion.Header>
                  <Accordion.Body>
                    <div className="mb-3">
                      <label className="form-label">
                        Battery Capacity Option:
                      </label>
                      <select
                        className="form-select"
                        name="batteryCapacityOption"
                        value={formData.batteryCapacityOption}
                        onChange={handleChange}
                      >
                        <option value="default">Default (100 kWh)</option>
                        <option value="user">User-Defined</option>
                      </select>
                    </div>
                    {formData.batteryCapacityOption === "user" && (
                      <div className="mb-3">
                        <label className="form-label">
                          Battery Capacity (kWh):
                        </label>
                        <input
                          type="number"
                          className="form-control"
                          name="batteryCapacityValue"
                          value={formData.batteryCapacityValue}
                          onChange={handleChange}
                          step="1"
                          min="0"
                        />
                      </div>
                    )}
                    <div className="mb-3">
                      <label className="form-label">
                        Load Capacity Option:
                      </label>
                      <select
                        className="form-select"
                        name="loadCapacityOption"
                        value={formData.loadCapacityOption}
                        onChange={handleChange}
                      >
                        <option value="default">Default (1000 Kg)</option>
                        <option value="user">User-Defined</option>
                      </select>
                    </div>
                    {formData.loadCapacityOption === "user" && (
                      <div className="mb-3">
                        <label className="form-label">
                          Load Capacity (Kg):
                        </label>
                        <input
                          type="number"
                          className="form-control"
                          name="loadCapacityValue"
                          value={formData.loadCapacityValue}
                          onChange={handleChange}
                          step="1"
                          min="0"
                        />
                      </div>
                    )}
                    <div className="mb-3">
                      <label className="form-label">
                        Charging Rate Option:
                      </label>
                      <select
                        className="form-select"
                        name="chargingRateOption"
                        value={formData.chargingRateOption}
                        onChange={handleChange}
                      >
                        <option value="default">Default (0.2)</option>
                        <option value="user">User-Defined</option>
                      </select>
                    </div>
                    {formData.chargingRateOption === "user" && (
                      <div className="mb-3">
                        <label className="form-label">
                          Charging Rate :
                        </label>
                        <input
                          type="number"
                          className="form-control"
                          name="chargingRateValue"
                          value={formData.chargingRateValue}
                          onChange={handleChange}
                          step="0.01"
                          min="0"
                        />
                      </div>
                    )}
                  </Accordion.Body>
                </Accordion.Item>

                <Accordion.Item eventKey="6">
                  <Accordion.Header>
                    <span className="accordion-icon">⏰</span>
                    Tour Time Limit
                  </Accordion.Header>
                  <Accordion.Body>
                    <div className="mb-3">
                      <label className="form-label">
                        Tour Time Limit (hours):
                      </label>
                      <input
                        type="number"
                        className="form-control"
                        name="tourTimeLimit"
                        value={formData.tourTimeLimit}
                        onChange={handleChange}
                        step="1"
                        min="0"
                      />
                    </div>
                  </Accordion.Body>
                </Accordion.Item>

          </Accordion>
        </div>

        {/* Sidebar Footer with Generate Button */}
        <div className="sidebar-footer">
          <button
            type="submit"
            className="btn btn-primary w-100"
            disabled={loading}
            onClick={handleSubmit}
          >
            {loading ? (
              <>
                <span 
                  className="spinner-border spinner-border-sm me-2" 
                  role="status" 
                  aria-hidden="true" 
                />
                Generating...
              </>
            ) : (
              "Generate Instance"
            )}
          </button>
        </div>
      </div>

      {/* RIGHT MAIN CONTENT AREA */}
      <div className="main-content position-relative">
        {/* Map Toggle Buttons - Only show if results exist */}
        {resultLinks && (
          <div className="map-toggle-container">
            <button
              className={`map-toggle-btn ${!showingResultMap ? 'active' : ''}`}
              onClick={() => setShowingResultMap(false)}
            >
              Input Map
            </button>
            <button
              className={`map-toggle-btn ${showingResultMap ? 'active' : ''}`}
              onClick={() => setShowingResultMap(true)}
            >
              Result Map
            </button>
          </div>
        )}

        {/* Map Container */}
        <div className="map-container">
          {!showingResultMap ? (
            <div className="map-wrapper">
              <MyMap onCoordinatesSelected={handleCoordinatesSelected} />
            </div>
          ) : (
            resultLinks?.htmlVisualization && (
              <iframe
                key={generatedKey}
                src={resultLinks.htmlVisualization}
                title="Instance Visualization"
                className="result-iframe"
              />
            )
          )}
        </div>

        {/* Results Drawer - Slides up from bottom */}
        {showResultsDrawer && (summaryData || responseMessage) && (
          <div className={`results-drawer ${isDrawerCollapsed ? 'collapsed' : ''}`}>
            <div className="results-drawer-header">
              <div 
                className="d-flex align-items-center" 
                style={{ cursor: 'pointer', flex: 1 }}
                onClick={() => setIsDrawerCollapsed(!isDrawerCollapsed)}
              >
                <h6 className="mb-0 me-2">Results & Statistics</h6>
                <button
                  className="results-drawer-toggle"
                  onClick={(e) => {
                    e.stopPropagation();
                    setIsDrawerCollapsed(!isDrawerCollapsed);
                  }}
                  aria-label={isDrawerCollapsed ? "Expand drawer" : "Collapse drawer"}
                >
                  {isDrawerCollapsed ? '▲' : '▼'}
                </button>
              </div>
              <button
                className="results-drawer-close"
                onClick={() => setShowResultsDrawer(false)}
                aria-label="Close drawer"
              >
                ×
              </button>
            </div>
            {!isDrawerCollapsed && (
              <div className="results-drawer-content">
                {/* Error/Info Message */}
                {responseMessage && (
                  <div 
                    className={`alert ${
                      responseMessage.includes('⚠️') || 
                      responseMessage.includes('INSUFFICIENT') || 
                      responseMessage.includes('ASSIGNMENT ERROR') ||
                      responseMessage.includes('COORDINATE VALIDATION ERROR') ||
                      responseMessage.includes('SOLUTIONS')
                        ? 'alert-danger' 
                        : 'alert-info'
                    }`}
                    style={{ 
                      whiteSpace: 'pre-wrap',
                      fontSize: '0.875rem',
                      lineHeight: '1.5',
                      marginBottom: '1rem'
                    }}
                    dangerouslySetInnerHTML={{ __html: formatErrorMessage(responseMessage) }}
                  />
                )}

                {/* Download Buttons */}
                {resultLinks && (
                  <div className="download-buttons-container">
                    <a
                      href={resultLinks.dataset}
                      download
                      className="btn btn-success btn-sm"
                    >
                      Download Dataset
                    </a>
                    {resultLinks.htmlVisualization && (
                      <>
                        <a
                          href={resultLinks.htmlVisualization}
                          className="btn btn-info btn-sm"
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          View HTML
                        </a>
                        <a
                          href={resultLinks.htmlVisualization}
                          download={`${formData.instanceName || 'instance'}.html`}
                          className="btn btn-info btn-sm"
                        >
                          Download HTML
                        </a>
                      </>
                    )}
                    <a
                      href={`${resultLinks.dataset.replace(".txt", "_summary.txt")}`}
                      download
                      className="btn btn-warning btn-sm"
                    >
                      Download Summary
                    </a>
                  </div>
                )}

                {/* Statistics and Charts - Side by Side Layout */}
                <div className="row">
                  {/* Left Column: Statistics */}
                  <div className="col-md-4">
                    {generalStats.length > 0 && (
                      <div>
                        <h6 className="text-secondary mb-3">Instance Statistics:</h6>
                        <ul className="list-group">
                          {generalStats.map((stat) => (
                            <li
                              key={stat.label}
                              className="list-group-item d-flex justify-content-between"
                            >
                              <strong>{stat.label}:</strong> {stat.value}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>

                  {/* Right Column: Chart */}
                  <div className="col-md-8">
                    {roadTypeChartData && (
                      <div className="chart-container">
                        <h6 className="text-secondary mb-3">Road Type Distribution:</h6>
                        <Bar data={roadTypeChartData} options={{ responsive: true, maintainAspectRatio: false }} />
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Info Modal */}
      <Modal
        show={showInfoModal}
        onHide={() => setShowInfoModal(false)}
        centered
        size="lg"
      >
        <Modal.Header closeButton className="modal-clean-header">
          <Modal.Title className="w-100 text-center">EVRPGen</Modal.Title>
        </Modal.Header>
        <Modal.Body className="about-modal-body">
          {/* Description */}
          <p className="lead text-muted text-center mb-2">
            A tool for generating realistic problem instances of the Electric Vehicle Routing Problem with Road Junctions and Road Types.
          </p>
          <p className="text-center text-muted small mb-5">
          A web-based instance generator for Electric Vehicle Routing Problems (EVRP) with Road Junctions and Road Types, using OpenStreetMap data. Users define an area, specify network components (depots, customers, charging stations, junctions), and customize vehicle parameters. 
          </p>

          {/* The Research Team */}
          <div className="mb-5">
            <h5 className="text-center mb-4">The Research Team</h5>
            <div className="row g-3">
              {/* Mehmet Anil AKBAY */}
              <div className="col-6">
                <div className="team-member-card text-center">
                  <img
                    src="/photos/photo_mehmet.jpg"
                    alt="Mehmet Anil AKBAY"
                    className="team-avatar"
                  />
                  <h6 className="mt-2 mb-1">Mehmet Anil AKBAY</h6>
                  <p className="team-role mb-1">Developer & Researcher</p>
                  <a
                    href="https://www.iiia.csic.es/es/people/person/?person_id=139"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-decoration-none small"
                    style={{ color: '#0d6efd' }}
                  >
                    IIIA-CSIC Profile
                  </a>
                </div>
              </div>

              {/* Christian Blum */}
              <div className="col-6">
                <div className="team-member-card text-center">
                  <img
                    src="/photos/cblum_small.png"
                    alt="Christian Blum"
                    className="team-avatar"
                  />
                  <h6 className="mt-2 mb-1">Christian Blum</h6>
                  <p className="team-role mb-1">Scientific Advisor</p>
                  <a
                    href="https://www.iiia.csic.es/~christian.blum/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-decoration-none small"
                    style={{ color: '#0d6efd' }}
                  >
                    IIIA-CSIC Profile
                  </a>
                </div>
              </div>
            </div>
          </div>

          {/* Resources & Documentation */}
          <div className="mb-5">
            <h5 className="text-center mb-4">Resources & Documentation</h5>
            <div className="row g-3">
              {/* GitHub Repository Card */}
              <div className="col-4">
                <div className="resource-list-item">
                  <div className="d-flex align-items-start mb-2">
                    <span className="resource-icon me-2">
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                      </svg>
                    </span>
                    <h6 className="mb-0">Repository</h6>
                  </div>
                  <p className="text-muted small mb-3">Source code and documentation.</p>
                  <a
                    href="https://github.com/manilakbay/InstanceGenerator.git"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="resource-badge"
                  >
                    View Link
                  </a>
                </div>
              </div>

              {/* Tutorial Card */}
              <div className="col-4">
                <div className="resource-list-item">
                  <div className="d-flex align-items-start mb-2">
                    <span className="resource-icon me-2">📚</span>
                    <h6 className="mb-0">Tutorial</h6>
                  </div>
                  <p className="text-muted small mb-3">Step-by-step guide to generating instances.</p>
                  <a
                    href="https://github.com/manilakbay/InstanceGenerator/tree/main/tutorial"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="resource-badge"
                  >
                    View Link
                  </a>
                </div>
              </div>

              {/* The Paper Card */}
              <div className="col-4">
                <div className="resource-list-item">
                  <div className="d-flex align-items-start mb-2">
                    <span className="resource-icon me-2">
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" xmlns="http://www.w3.org/2000/svg">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                        <line x1="16" y1="13" x2="8" y2="13"></line>
                        <line x1="16" y1="17" x2="8" y2="17"></line>
                        <polyline points="10 9 9 9 8 9"></polyline>
                      </svg>
                    </span>
                    <h6 className="mb-0">The Paper</h6>
                  </div>
                  <p className="text-muted small mb-3">Scientific publication and research paper.</p>
                  <a
                    href="https://www.sciencedirect.com/science/article/pii/S2665963825000387"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="resource-badge"
                  >
                    View Link
                  </a>
                </div>
              </div>
            </div>
          </div>

          {/* Contact */}
          <div className="mb-4 text-center">
            <p className="mb-0">
              <a href="mailto:makbay@iiia.csic.es" className="text-decoration-none">
                📧 makbay@iiia.csic.es
              </a>
            </p>
          </div>

          {/* Project Affiliations */}
          <div className="modal-logos">
            <img
              src="/logos/iiia_csic_logo.png"
              alt="IIIA-CSIC Logo"
              className="affiliation-logo"
            />
            <img
              src="/logos/miciu-aei.png"
              alt="MICIU AEI Logo"
              className="affiliation-logo"
            />
          </div>
        </Modal.Body>
        <Modal.Footer>
          <Button 
            variant="outline-primary" 
            onClick={() => {
              setShowInfoModal(false);
              // Small delay to ensure modal is closed before tour starts
              setTimeout(() => {
                setRunTour(true);
              }, 300);
            }}
          >
            Start Guided Tour
          </Button>
          <Button variant="secondary" onClick={() => setShowInfoModal(false)}>
            Close
          </Button>
        </Modal.Footer>
      </Modal>

      {/* Guided Tour */}
      <Joyride
        steps={tourSteps}
        run={runTour}
        continuous={true}
        showSkipButton={true}
        showProgress={true}
        callback={handleJoyrideCallback}
        styles={{
          options: {
            zIndex: 10000,
            primaryColor: '#0d6efd',
          },
          tooltip: {
            borderRadius: 8,
            fontSize: 14,
          },
          tooltipContainer: {
            textAlign: 'left',
          },
          buttonNext: {
            backgroundColor: '#0d6efd',
            fontSize: 14,
            padding: '8px 16px',
            borderRadius: 6,
          },
          buttonBack: {
            marginRight: 10,
            fontSize: 14,
            padding: '8px 16px',
            borderRadius: 6,
          },
          buttonSkip: {
            fontSize: 14,
            color: '#6c757d',
          },
        }}
        locale={{
          back: 'Back',
          close: 'Close',
          last: 'Finish',
          next: 'Next',
          skip: 'Skip Tour',
        }}
      />
    </div>
  );
}

export default App;
