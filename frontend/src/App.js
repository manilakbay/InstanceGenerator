import React, { useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { Bar } from "react-chartjs-2";
import Chart from "chart.js/auto";
import MyMap from "./MyMap";
import Accordion from "react-bootstrap/Accordion";
import Tabs from "react-bootstrap/Tabs";
import Tab from "react-bootstrap/Tab";

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

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResponseMessage("");
    setResultLinks(null);
    setSummaryData(null);

    try {
      const response = await fetch(`${API_BASE_URL}/generate-instance`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
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

  return (
    <div className="App">
      {/* Header */}
      <div className="d-flex align-items-center justify-content-center px-4 py-3 bg-light">
        <img
          src="/logos/iiia_csic_logo.png"
          alt="Institution Logo"
          height="50"
          className="me-3"
        />
        <h1 className="text-primary mb-0" style={{ fontSize: "1.8rem" }}>
          EVRPGen: Instance Generator for the Electric Vehicle Routing Problems with Road Junctions and Road Types
        </h1>
      </div>

      <div className="container mt-4">
        {/* First Row: Configuration Panel & Map */}
        <div className="row">
          {/* Left Column: Collapsible Configuration Panel (sticky sidebar) */}
          <div className="col-md-4 mb-4">
            <div className="card shadow p-4 h-100 sticky-top" style={{ top: "20px" }}>
              <Accordion defaultActiveKey="0" flush>
                <Accordion.Item eventKey="0">
                  <Accordion.Header>General Settings</Accordion.Header>
                  <Accordion.Body>
                    <div className="mb-3">
                      <label className="form-label">Select Area Method:</label>
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

                <Accordion.Item eventKey="2">
                  <Accordion.Header>Node Settings</Accordion.Header>
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


                <Accordion.Item eventKey="3">
                  <Accordion.Header>Customer Demand</Accordion.Header>
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
                <Accordion.Item eventKey="4">
                  <Accordion.Header>Service Time</Accordion.Header>
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
                <Accordion.Item eventKey="5">
                  <Accordion.Header>Vehicle Configuration</Accordion.Header>
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
                  <Accordion.Header>Tour Time Limit</Accordion.Header>
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
              <div className="mt-3">
                <button
                  type="submit"
                  className="btn btn-primary w-100"
                  disabled={loading}
                  onClick={handleSubmit}
                >
                  {loading ? (
                    <>
                      {/* Bootstrap Spinner */}
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
          </div>

          {/* Right Column: Map */}
          <div className="col-md-8 mb-4">
            <div className="card shadow p-4 h-100">
              <h5 className="mb-3 text-primary">Map</h5>
              {/* </div><div style={{ width: "100%", height: "100%", position: "relative" }} */}
              <div style={{width: "100%", height: "800px", border: "1px solid #ccc", display: "block", margin: "0 auto",}}>
                <MyMap onCoordinatesSelected={handleCoordinatesSelected} />
              </div>
            </div>
          </div>
        </div>

        
        {/* Second Row: Results in Tabs */}
        <div className="row">
          <div className="col-md-12">
            <div className="card shadow p-4">
              <Tabs
                defaultActiveKey="generated"
                id="results-tabs"
                className="mb-3"
                onSelect={(key) => {
                  if (key === "generated") {
                    setGeneratedKey(Date.now());
                  }
                }}
              >
                {/* Swap the order: Generated Instance tab appears first */}
                <Tab eventKey="generated" title="Generated Instance">
                  {responseMessage && (
                    <div 
                      className={`alert ${
                        responseMessage.includes('⚠️') || 
                        responseMessage.includes('INSUFFICIENT') || 
                        responseMessage.includes('ASSIGNMENT ERROR') ||
                        responseMessage.includes('SOLUTIONS')
                          ? 'alert-danger' 
                          : 'alert-info'
                      }`}
                      style={{ 
                        whiteSpace: 'pre-wrap',
                        fontSize: '0.95rem',
                        lineHeight: '1.6',
                        marginBottom: '1rem'
                      }}
                      dangerouslySetInnerHTML={{ __html: formatErrorMessage(responseMessage) }}
                    />
                  )}
                  <div className="text-center mb-3">
                    {resultLinks && (
                      <>
                        <a
                          // href={resultLinks.dataset}
                          // className="btn btn-success me-2"
                          // target="_blank"
                          // rel="noopener noreferrer"

                          href={resultLinks.dataset}
                          download
                          className="btn btn-success me-2"
                        >
                          Download Dataset
                        </a>
                        {/* {resultLinks?.htmlVisualization && (
                          <a
                            // href={resultLinks.htmlVisualization}
                            // className="btn btn-info me-2"
                            // target="_blank"
                            // rel="noopener noreferrer"
                            href={resultLinks.htmlVisualization}
                            download
                            className="btn btn-info me-2"
                          >
                            Download HTML Visualization
                          </a>
                        )} */}
                           {resultLinks.htmlVisualization && (
                            <>
                              {/* view inline */}
                              <a
                                href={resultLinks.htmlVisualization}
                                className="btn btn-info me-2"
                                target="_blank"
                                rel="noopener noreferrer"
                              >
                                View HTML Visualization
                              </a>
                              {/* download */}
                              <a
                                href={resultLinks.htmlVisualization}
                                download={`${formData.instanceName || 'instance'}.html`}
                                className="btn btn-info me-2"
                              >
                                Download HTML Visualization
                              </a>
                            </>
                           )}
                        <a
                          // href={`${resultLinks.dataset.replace(
                          //   ".txt",
                          //   "_summary.txt"
                          // )}`}
                          // className="btn btn-warning"
                          // target="_blank"
                          // rel="noopener noreferrer"
                          href={`${resultLinks.dataset.replace(".txt", "_summary.txt")}`}
                          download
                          className="btn btn-warning"
                        >
                          Download Summary File
                        </a>
                      </>
                    )}
                  </div>
                  <div className="text-center">
                    {resultLinks?.htmlVisualization && (
                      <iframe
                        key={generatedKey}
                        src={resultLinks.htmlVisualization}
                        title="Instance Visualization"
                        style={{
                          width: "1200px",
                          height: "1200px",
                          border: "1px solid #ccc",
                          display: "block",
                          margin: "0 auto",
                        }}
                      />
                    )}
                  </div>
                </Tab>
                <Tab eventKey="stats" title="Statistics & Road Type Distribution">
                  {generalStats.length > 0 && (
                    <div className="mb-4">
                      <h6 className="text-secondary">Instance Statistics:</h6>
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
                  {roadTypeChartData && (
                    <div className="mb-4">
                      <h6 className="text-secondary">Road Type Distribution:</h6>
                      <Bar data={roadTypeChartData} options={{ responsive: true }} />
                    </div>
                  )}
                </Tab>
              </Tabs>
            </div>
          </div>
        </div>
      </div>
      {/* New Footer Section */}
      <footer className="bg-dark text-white mt-4">
        <div className="container py-4">
          <div className="row">
            <div className="col-lg-4 col-md-12 mb-3 mb-lg-0">
              <h5 className="text-uppercase">Authors</h5>
              <p className="mb-0">Mehmet Anil AKBAY</p>
              <p className="mb-0">Christian Blum</p>
            </div>
            <div className="col-lg-4 col-md-6 mb-3 mb-md-0">
              <h5 className="text-uppercase">Useful Links</h5>
              <ul className="list-unstyled">
                <li>
                  <a href="https://github.com/manilakbay/InstanceGenerator.git" className="text-white text-decoration-none">
                    Github Repository
                  </a>
                </li>
                <li>
                  <a href="https://github.com/manilakbay/InstanceGenerator/tree/main/tutorial" className="text-white text-decoration-none">
                    Tutorial
                  </a>
                </li>
              </ul>
            </div>
            <div className="col-lg-4 col-md-6 mb-3 mb-md-0">
              <h5 className="text-uppercase">Contact</h5>
              <p className="mb-0">Email: <a href="mailto:makbay@iiia.csic.es" className="text-white text-decoration-none">makbay@iiia.csic.es</a></p>
            </div>
          </div>
        </div>
        <div className="bg-secondary text-center py-2">
          <img
            src="/logos/iiia_csic_logo.png"
            alt="Institution Logo"
            height="30"
            className="me-2"
          />
          © {new Date().getFullYear()} Instituto de Investigación en Inteligencia Artificial (IIIA-CSIC). All rights reserved.

          <img
            src="/logos/miciu-aei.png"
            alt="Agencia Logo"
            height="30"
            className="me-2"
          />

        </div>
      </footer>


      {/* <footer className="bg-dark text-white mt-4">
        <div className="container py-4">
          <div className="row">
            <div className="col-lg-4 col-md-12 mb-3 mb-lg-0">
              <h5 className="text-uppercase">Authors</h5>
              <p className="mb-0">Mehmet Anil AKBAY, Christian Blum</p>
            </div>
            <div className="col-lg-4 col-md-6 mb-3 mb-md-0">
              <h5 className="text-uppercase">Useful Links</h5>
              <ul className="list-unstyled">
                <li>
                  <a href="https://github.com/manilakbay/InstanceGenerator.git" className="text-white text-decoration-none">
                    Github Repository
                  </a>
                </li>
                <li>
                  <a href="https://github.com/manilakbay/InstanceGenerator/tree/main/tutorial" className="text-white text-decoration-none">
                    Tutorial
                  </a>
                </li>
              </ul>
            </div>
            <div className="col-lg-4 col-md-6 mb-3 mb-md-0">
              <h5 className="text-uppercase">Contact</h5>
              <p className="mb-0">
                Email:{" "}
                <a href="mailto:makbay@iiia.csic.es" className="text-white text-decoration-none">
                  makbay@iiia.csic.es
                </a>
              </p>
            </div>
          </div>
        </div> */}

        {/* Modified this area: now uses a flexbox that justifies content between the left and right sides. */}
        {/* <div className="bg-secondary py-2 d-flex justify-content-between align-items-center">
          <div>
            <img
              src="/logos/iiia_csic_logo.png"
              alt="Institution Logo"
              height="30"
              className="me-2"
            />
            © {new Date().getFullYear()} Your Institution Name. All rights reserved.
          </div> */}

          {/* New logo on the right side */}
          {/* <div>
            <img
              src="/logos/miciu-aei.png"
              alt="MICIU AEI Logo"
              height="30"
            />
          </div>
        </div>
      </footer> */}


    </div>
  );
}

export default App;
