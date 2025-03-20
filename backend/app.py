from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import subprocess
import os

app = Flask(
    __name__,
    static_folder="frontend_dist",      # The folder with your React build
    static_url_path=""                  # Serve from root path
)
CORS(app)

#@app.route('/generate-instance', methods=['POST'])

#Catch-all route for React Single-Page App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_frontend(path):
    # if the requested file exists in frontend_dist, serve it
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        # otherwise, serve index.html (the React app)
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/generate-instance', methods=['POST'])
def generate_instance():
    data = request.json

    # General parameters:
    area_selection_method = data.get('areaSelectionMethod')  # address or coordinates
    area_name = data.get('area')
    coordinates = data.get('coordinates')
    show_arrows = data.get('show_arrows', False)
    num_depots = data.get('numDepots', 1)
    num_customers = data.get('numCustomers', 10)
    num_cs = data.get('numChargingStations', 5)
    simplify = data.get('simplify', False)
    customer_tag_value = data.get('customerChoice', "supermarket")  # Default to 'supermarket'
    instance_name = data.get('instanceName', 'instance')

    # Demand & service-time
    demand_option = data.get('demandOption', 'default')
    demand_constant = data.get('demandConstant', 100)
    demand_range_min = data.get('demandRangeMin', 50)
    demand_range_max = data.get('demandRangeMax', 240)
    service_time_option = data.get('serviceTimeOption', 'default')
    service_time_constant = data.get('serviceTimeConstant', 0.5)
    service_time_min = data.get('serviceTimeMin', 0.1)
    service_time_max = data.get('serviceTimeMax', 1.0)

    # Vehicle config
    battery_capacity_option = data.get('batteryCapacityOption', 'default')
    battery_capacity_value = data.get('batteryCapacityValue', 100.0)
    load_capacity_option = data.get('loadCapacityOption', 'default')
    load_capacity_value = data.get('loadCapacityValue', 1000.0)
    charging_rate_option = data.get('charginRateOption', 'default')
    charging_rate_value = data.get('chargingRateValue', 0.2)
 
    # Time Limit
    tour_time_limit = data.get('tourTimeLimit', 8)  # default 8 if not provided

    # Seed for the random number generator
    random_seed = data.get('randomSeed', 42)  # default = 42 if not provided


    try:
        # Base directory for outputs
        output_base = "created_datasets"
        if not os.path.exists(output_base):
            os.makedirs(output_base)

        # Prepare command for subprocess
        command = ["python", "create_instance.py"]
        
        if area_selection_method == "address":
            command.extend(["--area_name", area_name])
        elif area_selection_method == "coordinates":
            command.extend(["--coordinates", coordinates])
        else:
            raise ValueError("Invalid area selection method. Must be 'address' or 'coordinates'.")

        command.extend([
            "--num_depots", str(num_depots),
            "--num_customers", str(num_customers),
            "--num_cs", str(num_cs),
            "--customer_tag", customer_tag_value,
            "--instance_name", instance_name,  
            "--demand_option", demand_option,
            "--demand_constant", str(demand_constant),
            "--demand_range_min", str(demand_range_min),
            "--demand_range_max", str(demand_range_max),
            "--service_time_option", service_time_option,
            "--service_time_constant", str(service_time_constant),
            "--service_time_min", str(service_time_min),
            "--service_time_max", str(service_time_max),
            "--battery_capacity_option", battery_capacity_option,
            "--battery_capacity_value", str(battery_capacity_value),
            "--load_capacity_option", load_capacity_option,
            "--load_capacity_value", str(load_capacity_value),
            "--charging_rate_option", charging_rate_option,
            "--charging_rate_value", str(charging_rate_value),
            "--tour_time_limit", str(tour_time_limit),
            "--random_seed", str(random_seed)
        ])

        if simplify:
            command.append("--simplify")
        
        if show_arrows:
            command.append("--show_arrows")

        print("Command: ", command)

        # Execute the script
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if process.returncode != 0:
            raise RuntimeError(f"Error: {process.stderr.strip()}")
        
        # Extract location message from stdout
        stdout_output = process.stdout
        location_message = "Instance generated successfully."  # Default fallback
        for line in stdout_output.split("\n"):
            if "Successfully found" in line or "The rest selected randomly" in line:
                location_message = line.strip()
                break

        # Get custom instance name if provided
        instance_name = data.get('instanceName', '').strip()
        if instance_name:
            # Use the custom name for both the directory and the dataset name.
            area_dir = instance_name.replace(" ", "_")
            dataset_name = area_dir
        else:
            # Default naming logic when no instance name is provided.
            if area_selection_method == "address":
                area_dir = f"{area_name.split(',')[0].replace(' ', '_')}_{num_customers}"
                dataset_name = f"{area_dir}_{'simplified' if simplify else 'not_simplified'}"
            elif area_selection_method == "coordinates":
                coordinates_str = coordinates.replace(',', '_')
                area_dir = f"{coordinates_str}_{num_customers}"
                dataset_name = f"{coordinates_str}_{num_customers}_{'simplified' if simplify else 'not_simplified'}"


        output_dir = os.path.join(output_base, area_dir)

        dataset_path = os.path.join(output_dir, f"{dataset_name}.txt")
        summary_path = os.path.join(output_dir, f"{dataset_name}_summary.txt")

        print("Dataset Path: ", dataset_path)
        print("Dataset Name: ", dataset_name)

        # Verify dataset & summary
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"Dataset file {dataset_path} is missing.")
        if not os.path.isfile(summary_path):
            raise FileNotFoundError(f"Summary file {summary_path} is missing.")

        # Parse summary
        summary_data = {}
        with open(summary_path, "r") as file:
            for line in file:
                if ":" in line:
                    key, value = line.split(":", 1)
                    summary_data[key.strip()] = value.strip()

        road_color_map = {
            "motorway": "red", "motorway_link": "darkred", "trunk": "orange", "trunk_link": "darkorange",
            "primary": "yellow", "primary_link": "gold", "secondary": "green", "secondary_link": "darkgreen",
            "tertiary": "blue", "tertiary_link": "darkblue", "residential": "purple", "service": "gray",
            "unclassified": "brown", "road": "pink", "unknown": "black"
        }

        response = {
            "message": f"Instance generated successfully for {area_name if area_selection_method == 'address' else coordinates}. {location_message}",
            "dataset_url": f"/download/{area_dir}/{dataset_name}.txt",
            "html_visualization_url": f"/download/{area_dir}/{dataset_name}.html",
            "summary_data": summary_data,
            "road_colors": road_color_map
        }


        response = {
            "message": f"Instance generated for the selected area. {location_message}",
            "dataset_url": f"/download/{area_dir}/{dataset_name}.txt",
            "html_visualization_url": f"/download/{area_dir}/{dataset_name}.html",
            "summary_data": summary_data,
            "road_colors": road_color_map
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"message": f"Error generating instance: {str(e)}"}), 500

@app.route('/download/<path:subdir>/<filename>', methods=['GET'])
def download_file(subdir, filename):
    output_dir = os.path.join("created_datasets", subdir)
    file_path = os.path.join(output_dir, filename)
    
    if not os.path.exists(file_path):
        return jsonify({"message": "File not found."}), 404
    
    if filename.endswith(".html"):
        return send_file(file_path, mimetype="text/html")
    
    return send_from_directory(output_dir, filename)

# @app.route('/', methods=['GET'])
# def home():
#     return jsonify({"message": "Welcome to the Instance Generator API!"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
