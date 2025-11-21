from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import subprocess
from subprocess import TimeoutExpired
import os
from datetime import datetime
import logging

app = Flask(
    __name__,
    static_folder="frontend_dist",      # The folder with your React build
    static_url_path=""                  # Serve from root path
)

# Configure CORS properly - restrict to authorized origins
allowed_origins_str = os.environ.get(
    'ALLOWED_ORIGINS',
    'http://localhost:3000,http://localhost:5000'  # Default for development
)
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(',')]

CORS(app, resources={
    r"/*": {
        "origins": allowed_origins,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "max_age": 3600
    }
})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#@app.route('/generate-instance', methods=['POST'])

# Health check endpoint for monitoring and load balancers
@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    """
    try:
        # Check if critical directories exist
        script_dir = os.path.dirname(os.path.abspath(__file__))
        created_datasets_dir = os.path.join(script_dir, "created_datasets")
        if not os.path.exists(created_datasets_dir):
            return jsonify({
                "status": "degraded",
                "message": "Output directory missing",
                "timestamp": datetime.now().isoformat()
            }), 503
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 503

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

    # Generation mode: 'random' or 'upload'
    generation_mode = data.get('generationMode', 'random')  # default to 'random' for backward compatibility

    # Capture user information from request
    # Get IP address (handles proxy/load balancer)
    user_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('HTTP_X_REAL_IP', request.remote_addr))
    if user_ip:
        # If multiple IPs (from proxies), take the first one
        user_ip = user_ip.split(',')[0].strip()
    else:
        user_ip = request.remote_addr or 'Unknown'
    
    user_agent = request.headers.get('User-Agent', 'Unknown')
    user_referer = request.headers.get('Referer', 'Direct')

    uploaded_file_path = None
    try:
        # Get the directory where app.py is located (this is where create_instance.py also is)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Base directory for outputs - use absolute path to ensure consistency
        output_base = os.path.join(script_dir, "created_datasets")
        if not os.path.exists(output_base):
            os.makedirs(output_base)

        # Handle file upload if in upload mode
        if generation_mode == 'upload' and data.get('uploadedFileContent'):
            # Create temp_uploads directory if it doesn't exist
            temp_uploads_dir = os.path.join(script_dir, "temp_uploads")
            if not os.path.exists(temp_uploads_dir):
                os.makedirs(temp_uploads_dir)
            
            # Generate unique filename based on instance name and timestamp
            safe_instance_name = instance_name.replace(" ", "_") if instance_name else "instance"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            uploaded_file_path = os.path.join(temp_uploads_dir, f"{safe_instance_name}_{timestamp}_coords.csv")
            
            # Write uploaded file content to temporary file
            try:
                with open(uploaded_file_path, 'w', encoding='utf-8') as f:
                    f.write(data['uploadedFileContent'])
                logger.info(f"Saved uploaded file to: {uploaded_file_path}")
            except Exception as e:
                logger.error(f"Error saving uploaded file: {str(e)}")
                return jsonify({
                    "message": f"Error saving uploaded file: {str(e)}",
                    "error": True,
                    "type": "file_error"
                }), 500

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
            "--random_seed", str(random_seed),
            "--user_ip", user_ip,
            "--user_agent", user_agent,
            "--user_referer", user_referer
        ])

        if simplify:
            command.append("--simplify")
        
        if show_arrows:
            command.append("--show_arrows")

        # Add uploaded file path if in upload mode
        if generation_mode == 'upload' and uploaded_file_path:
            command.extend(["--uploaded_file_path", uploaded_file_path])

        print("Command: ", command)

        # Execute the script with timeout
        timeout_seconds = int(os.environ.get('INSTANCE_GENERATION_TIMEOUT', 1800))  # 30 min default
        
        try:
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
                cwd=script_dir  # Run from the same directory as create_instance.py
            )
        except TimeoutExpired:
            error_msg = (
                f"Instance generation timed out after {timeout_seconds} seconds. "
                f"The requested instance may be too large or complex. "
                f"Please try with a smaller area or fewer nodes."
            )
            logger.error(f"Subprocess timeout: {error_msg}")
            return jsonify({
                "message": error_msg,
                "error": True,
                "type": "timeout_error"
            }), 408

        if process.returncode != 0:
            error_output = process.stderr.strip()
            # Extract the actual error message (remove traceback)
            error_lines = error_output.split('\n')
            # Look for ValueError or other meaningful error messages
            clean_error = error_output
            for line in reversed(error_lines):
                if 'ValueError:' in line or 'Error:' in line or '⚠️' in line:
                    # Found the actual error message, extract from here
                    error_start = error_output.find(line)
                    clean_error = error_output[error_start:].strip()
                    break
            # If error contains the warning emoji, it's a user-friendly error
            if '⚠️' in clean_error:
                # Extract just the message part after ValueError:
                if 'ValueError:' in clean_error:
                    clean_error = clean_error.split('ValueError:', 1)[-1].strip()
            else:
                # Fallback: try to get last meaningful line
                clean_error = error_lines[-1] if error_lines else "Unknown error occurred"
            
            raise RuntimeError(clean_error)
        
        # Extract location message from stdout
        stdout_output = process.stdout
        location_message = "Instance generated successfully."  # Default fallback
        for line in stdout_output.split("\n"):
            if "Successfully found" in line or "The rest selected randomly" in line or "Using" in line and "from uploaded file" in line:
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
            # IMPORTANT: This must match the logic in create_instance.py exactly
            if area_selection_method == "address":
                area_dir = f"{area_name.split(',')[0].replace(' ', '_')}_{num_customers}"
                dataset_name = f"{area_dir}_{'simplified' if simplify else 'not_simplified'}"
            elif area_selection_method == "coordinates":
                coordinates_str = coordinates.replace(',', '_')
                # Match create_instance.py logic: if coordinates_str > 50 chars, use "instance_poly"
                if len(coordinates_str) > 50:
                    short_coords = "instance_poly"
                else:
                    short_coords = coordinates_str
                area_dir = f"{short_coords}_{num_customers}"
                dataset_name = f"{short_coords}_{num_customers}_{'simplified' if simplify else 'not_simplified'}"


        output_dir = os.path.join(output_base, area_dir)

        dataset_path = os.path.join(output_dir, f"{dataset_name}.txt")
        summary_path = os.path.join(output_dir, f"{dataset_name}_summary.txt")

        logger.info(f"Expected dataset path: {dataset_path}")
        logger.info(f"Expected summary path: {summary_path}")
        logger.info(f"Output directory: {output_dir}")
        
        # Log subprocess output for debugging
        if process.stdout:
            logger.info(f"Subprocess stdout (last 20 lines):\n{chr(10).join(process.stdout.split(chr(10))[-20:])}")
        if process.stderr:
            logger.warning(f"Subprocess stderr:\n{process.stderr}")

        # Wait a moment for file system to sync (in case of race condition)
        import time
        time.sleep(0.5)
        
        # Check if directory exists
        if not os.path.exists(output_dir):
            error_msg = (
                f"Output directory {output_dir} was not created. "
                f"The instance generation may have failed. "
                f"Please check the logs for more details."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # List files in directory for debugging
        if os.path.exists(output_dir):
            files_in_dir = os.listdir(output_dir)
            logger.info(f"Files in output directory: {files_in_dir}")

        # Verify dataset & summary
        if not os.path.isfile(dataset_path):
            # Provide more helpful error message
            error_msg = (
                f"Dataset file {dataset_path} is missing.\n\n"
                f"Expected filename: {dataset_name}.txt\n"
                f"Output directory: {output_dir}\n"
                f"Files found in directory: {', '.join(os.listdir(output_dir)) if os.path.exists(output_dir) else 'Directory does not exist'}\n\n"
                f"The instance generation process may have failed. Please check the timing log file for details."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        if not os.path.isfile(summary_path):
            error_msg = (
                f"Summary file {summary_path} is missing.\n\n"
                f"Expected filename: {dataset_name}_summary.txt\n"
                f"Output directory: {output_dir}\n"
                f"Files found in directory: {', '.join(os.listdir(output_dir)) if os.path.exists(output_dir) else 'Directory does not exist'}"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

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
            "message": f"Instance generated for the selected area. {location_message}",
            "dataset_url": f"/download/{area_dir}/{dataset_name}.txt",
            "html_visualization_url": f"/download/{area_dir}/{dataset_name}.html",
            "summary_data": summary_data,
            "road_colors": road_color_map
        }

        return jsonify(response)

    except Exception as e:
        # Clean up temporary uploaded file if it exists (even on error)
        if uploaded_file_path and os.path.exists(uploaded_file_path):
            try:
                os.remove(uploaded_file_path)
                logger.info(f"Cleaned up temporary file after error: {uploaded_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Could not remove temporary file {uploaded_file_path}: {str(cleanup_error)}")
        
        error_msg = str(e)
        # Remove "Error: " prefix if present
        if error_msg.startswith("Error: "):
            error_msg = error_msg[7:]
        
        # Clean up any remaining traceback artifacts
        error_lines = error_msg.split('\n')
        clean_lines = []
        for line in error_lines:
            # Skip traceback lines (contain file paths or "Traceback")
            if '/app/' not in line and 'File "' not in line and 'Traceback' not in line and 'line ' not in line:
                clean_lines.append(line)
        
        error_msg = '\n'.join(clean_lines).strip()
        
        # Determine error type
        error_type = "validation_error" if "⚠️" in error_msg or "INSUFFICIENT" in error_msg or "SOLUTIONS" in error_msg else "generation_error"
        
        # Return structured error response
        return jsonify({
            "message": error_msg,
            "error": True,
            "type": error_type
        }), 500

# @app.route('/download/<path:subdir>/<filename>', methods=['GET'])
# def download_file(subdir, filename):
#     output_dir = os.path.join("created_datasets", subdir)
#     file_path = os.path.join(output_dir, filename)
    
#     if not os.path.exists(file_path):
#         return jsonify({"message": "File not found."}), 404
    
#     if filename.endswith(".html"):
#         return send_file(file_path, mimetype="text/html")
    
#     return send_from_directory(output_dir, filename)



@app.route('/download/<path:subdir>/<filename>', methods=['GET'])
def download_file(subdir, filename):
    # Get the directory where app.py is located to ensure consistent path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "created_datasets", subdir)
    file_path  = os.path.join(output_dir, filename)
    if not os.path.exists(file_path):
        return jsonify({"message": "File not found."}), 404

    if filename.lower().endswith(".html"):
        # serve inline so iframe can render
        return send_file(
            file_path,
            mimetype="text/html",
            as_attachment=False
        )

    # all other files (txt/summary) -> force download
    return send_file(
        file_path,
        as_attachment=True,
        download_name=filename
    )


# @app.route('/', methods=['GET'])
# def home():
#     return jsonify({"message": "Welcome to the Instance Generator API!"})

if __name__ == '__main__':
    # Never use debug=True in production
    # Gunicorn is used in production via Dockerfile, this is only for local development
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Safety check: never enable debug in production environment
    if os.environ.get('FLASK_ENV') == 'production':
        debug_mode = False
    
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
