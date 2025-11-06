# ------------------- STAGE 1: Build the React frontend -------------------
FROM node:18-alpine AS frontend-builder

# 1) Set working directory inside the container
WORKDIR /app/frontend

# 2) Accept build argument for API URL (defaults to same origin for production)
ARG REACT_APP_API_URL
ENV REACT_APP_API_URL=${REACT_APP_API_URL}

# 3) Copy ONLY the package.json and package-lock.json first to leverage Docker caching
COPY frontend/package*.json ./

# 4) Install dependencies for the frontend
RUN npm install

# 5) Copy the rest of the frontend source code
COPY frontend/ ./

# 6) Build the React app for production
RUN npm run build

# ------------------- STAGE 2: Set up Python + Flask backend -------------------
FROM python:3.9-slim

# 1) Create a working directory for your backend code
WORKDIR /app

# 2) Copy only the backend requirements file
COPY backend/requirements.txt /app

# 3) Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copy the entire backend code into /app
COPY backend/ /app

# 5) Copy the built React frontend (from Stage 1) into a folder in the backend
COPY --from=frontend-builder /app/frontend/build/ /app/frontend_dist/

# (Optional) You can copy logs or other data if needed; typically not needed for ephemeral containers.

# 6) Expose the port that Flask/Gunicorn will run on
EXPOSE 5000

# 7) Start the server using Gunicorn, binding to 0.0.0.0:5000
#    The "app:app" means "load 'app.py' and find the Flask instance named 'app'"
#CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "600", "app:app"]

