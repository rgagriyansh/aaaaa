# Math Document Converter Web App

A modern web application that converts images containing mathematical expressions into formatted Word documents using AI-powered text extraction and mathematical expression formatting.

## Features

- 🖼️ **Multiple Image Upload**: Upload one or multiple images at once
- 🤖 **AI-Powered OCR**: Uses Claude AI to extract text from images
- 📐 **Math Expression Detection**: Automatically identifies and formats mathematical expressions
- 📄 **Word Document Generation**: Creates professionally formatted Word documents
- 🎨 **Modern UI**: Beautiful, responsive web interface with drag-and-drop functionality
- 📱 **Mobile Friendly**: Works perfectly on desktop and mobile devices

## Supported Image Formats

- JPG/JPEG
- PNG
- GIF
- BMP
- TIFF
- WEBP

## Quick Start

### Option 1: Automatic Installation (Recommended)

```bash
python install.py
```

This will:
- Install all dependencies
- Upgrade to the latest Anthropic client
- Test the API connection
- Create necessary directories

### Option 2: Manual Installation

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install --upgrade anthropic
```

#### 2. Set Up API Key

Make sure you have a valid Claude API key. The application will use the key from your `main.py` file.

#### 3. Test API Connection

```bash
python test_api.py
```

#### 4. Run the Application

```bash
python run.py
```

The web application will be available at `http://localhost:5000`

## How to Use

1. **Open the Web App**: Navigate to `http://localhost:5000` in your browser
2. **Upload Images**: Drag and drop images or click to browse and select files
3. **Process**: Click "Process Images" to start the conversion
4. **Download**: Once processing is complete, download your Word documents

## Hosting Options

### Option 1: Local Development (Recommended for Testing)

```bash
python app.py
```

### Option 2: Production Server (Gunicorn)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Option 3: Cloud Hosting

#### Heroku
1. Create a `Procfile`:
```
web: gunicorn app:app
```

2. Deploy to Heroku:
```bash
heroku create your-app-name
git add .
git commit -m "Initial commit"
git push heroku main
```

#### Railway
1. Connect your GitHub repository to Railway
2. Railway will automatically detect the Python app and deploy it

#### Render
1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn app:app`

#### PythonAnywhere
1. Upload your files to PythonAnywhere
2. Install requirements: `pip install -r requirements.txt`
3. Configure WSGI file to point to your app
4. Set up your domain

### Option 4: Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:
```bash
docker build -t math-converter .
docker run -p 5000:5000 math-converter
```

## Environment Variables

For production deployment, consider setting these environment variables:

```bash
export FLASK_ENV=production
export SECRET_KEY=your-secret-key-here
```

## File Structure

```
├── app.py                 # Flask web application
├── main.py               # Core processing logic
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── templates/
│   └── index.html       # Web interface template
└── uploads/             # Temporary upload directory (auto-created)
```

## API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Upload and process images
- `GET /download/<filename>` - Download generated documents
- `GET /health` - Health check endpoint

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your Claude API key is valid and has sufficient credits
2. **File Upload Issues**: Check that files are under 16MB and in supported formats
3. **Processing Errors**: Ensure images contain clear, readable text
4. **Port Already in Use**: Change the port in `app.py` or kill the existing process

### Logs

The application logs important events. Check the console output for debugging information.

## Security Considerations

- Change the `SECRET_KEY` in production
- Implement rate limiting for production use
- Add authentication if needed
- Consider using HTTPS in production
- Regularly clean up temporary files

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License. 