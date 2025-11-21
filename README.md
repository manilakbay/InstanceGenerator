# A Web-Based Instance Generator for the Electric Vehicle Routing Problem with Road Junctions and Road Types

This repository provides a web application for generating instances of the **Electric Vehicle Routing Problem with Road Junctions and Road Types (EVRP-RJ-RT)**. It combines a **Flask** (Python) backend (with OSMnx for map data) and a **React** frontend to configure and visualize generated instances.

---

## How to Run the Instance Generator Using Docker (Locally)

### Prerequisites

- **Docker** must be installed and running on your system.
  - **Linux**: Ensure your user is in the `docker` group and that Docker is active.
  - **Windows/macOS**: Install **Docker Desktop**.

### 1. Build the Docker Image

1. **Clone** or download this repository.
2. **Open a terminal** in the repository's root directory (where the `Dockerfile` is located).
3. **Build** the Docker image by running:
   
   ```bash
   docker build -t evrp-generator .
   ```
   This command reads the `Dockerfile`, installs Python and Node.js dependencies, compiles the React frontend, and bundles everything with Flask & Gunicorn.

### 2. Run the Container

Once the image is built successfully, run:

```bash
docker run -d -p 5000:5000 evrp-generator
```

- `-d` starts the container in **detached mode**.
- `-p 5000:5000` maps port **5000** in the container to port **5000** on your local machine.
- The container will start **Gunicorn**, serving the Flask application.

### 3. Access the Web Interface

Open your browser and navigate to:

```
http://localhost:5000
```

You should see the **EVRP Instance Generator** interface. From here, you can:

- Select an area of interest
- Configure node counts
- Set demand/service-time options
- Customize vehicle configurations
- Click **"Generate Instance"** to build and visualize an instance

### 4. Outputs & Downloads

Once the generation process is complete, the **Results** tab provides options to download:

- **Dataset** (`.txt` file)
- **HTML Visualization** (`.html` file)
- **Summary File** (`_summary.txt` file)

These files are stored in the container’s `/app/created_datasets` directory. If you need them on your host machine, you can either **mount a volume** or **download them via the web interface**.

### 5. Stopping the Container

To stop the running container:

1. Find its **ID** or **name** using:
   
   ```bash
   docker ps
   ```

2. Stop it with:
   
   ```bash
   docker stop <container-id-or-name>
   ```

### Further Notes

- **Port Conflicts**: If port `5000` is already in use, modify the host port mapping, e.g., `-p 8081:5000`, then visit `http://localhost:8081`.
- **Editing Code**: If you modify the code or `requirements.txt`, you must rebuild the image by running:
  
  ```bash
  docker build -t evrp-generator .
  ```
- **Running Locally Without Docker**: If you prefer to run the application **natively** (without Docker), refer to the local setup instructions in `backend/requirements.txt`.

---

## Citation

If you use this EVRP Instance Generator in your research or projects, please consider citing:

```latex

@article{AKBAY2025100778,
title = {EVRPGen: A web-based instance generator for the electric vehicle routing problem with road junctions and road types},
journal = {Software Impacts},
volume = {25},
pages = {100778},
year = {2025},
issn = {2665-9638},
doi = {https://doi.org/10.1016/j.simpa.2025.100778},
url = {https://www.sciencedirect.com/science/article/pii/S2665963825000387},
author = {Mehmet Anil Akbay and Christian Blum},
keywords = {Electric vehicle routing problem (EVRP), Instance generation, OpenStreetMap (OSM), Road network analysis, Sustainable logistics, Web application},
abstract = {This paper presents a web-based instance generator for Electric Vehicle Routing Problems (EVRP) with Road Junctions and Road Types, using OpenStreetMap data. Users define an area, specify network components (depots, customers, charging stations, junctions), and customize vehicle parameters. The React-based frontend enables configuration, visualization, and queries, while the Flask backend processes road networks, classifies road types, and assigns demand and service times. A RESTful API ensures real-time instance generation. Generated instances can be downloaded as text-based datasets and interactive visualizations. The tool is open-source and contributes to the area of sustainable transportation by enabling scenario-based EVRP experimentation.}
}


@misc{akbay_blum_evrp2025,
  title   = {EVRPGen: A Web-Based Instance Generator for the Electric Vehicle Routing Problem with Road Junctions and Road Types},
  author  = {Mehmet Anil Akbay and Christian Blum},
  year    = {2025},
  howpublished = {\url{https://github.com/manilakbay/InstanceGenerator}}
}

```

For additional details, check the `/tutorial` folder or contact:

- **Mehmet Anil Akbay** ([makbay@iiia.csic.es](mailto:makbay@iiia.csic.es))
- **Christian Blum**

Enjoy exploring EVRP instances!

