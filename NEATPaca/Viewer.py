import os
import webbrowser
from threading import Timer, Thread
import http.server
import socketserver
import json

class Viewer:
    def __init__(self, dir_path="viz_output"):
        self.dir_path = dir_path
        self.server_thread = None
        self.master_chart_path = os.path.join(self.dir_path, "master_chart.html")
        self.Handler = http.server.SimpleHTTPRequestHandler
        self.PORT = 8000
        self.window_name = "NEATModelWindow"
        
        # Close any existing server and the associated browser tab
        self.stop_server()
        # self.close_existing_browser_tab()
        
        # Create the master HTML file
        self.create_master_html()
        
        # Start the server
        Timer(1, self.start_server).start()

    def sort_files(self, file):
        parts = file.split('_')
        if len(parts) > 2 and parts[1] == 'Gen' and parts[2].isdigit():
            return int(parts[2])
        return 0

    def create_master_html(self):
        # List all HTML files
        all_files = [f for f in os.listdir(self.dir_path) if f.endswith('.html')]
        
        # Ensure 'master_chart.html' is not included in the list
        all_files = [f for f in all_files if f != "master_chart.html"]
        
        # Extracting currencies and organizing files by currency
        files_by_currency = {}
        for file in all_files:
            parts = file.split('_')
            currency = parts[0]  # Assuming the first part of the filename is the currency
            if currency not in files_by_currency:
                files_by_currency[currency] = []
            files_by_currency[currency].append(file)
        
        # Sort files within each currency
        for currency in files_by_currency:
            files_by_currency[currency] = sorted(files_by_currency[currency], key=self.sort_files)
        
        # Generate options for the dropdown menu
        currency_options = ''.join(f'<option value="{currency}">{currency}</option>' for currency in sorted(files_by_currency.keys()))
        
        html_content = f"""
        <html>
        <head>
            <title>NEAT Model Trading Performance</title>
            <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600&display=swap" rel="stylesheet">
            <style>
                body {{font-family: 'Open Sans', sans-serif; margin: 0; padding: 0; display: flex; flex-direction: column; align-items: center;}}
                h3, p {{text-align: center;}}
                iframe {{width: 100%;}}
                .slider-container {{display: flex; align-items: center; justify-content: center; width:100%}}
                .slider {{width: 50%; margin-left: 20px; margin-right: 20px}}
                iframe {{height: 90%; border: none; margin-bottom: 10px;}}
            </style>
        </head>
        <body>
            <h3>NEAT Model Trading Performance</h3>
            <p>Select Currency:</p>
            <select id="currencySelect">
                {currency_options}
            </select>
            <p>Generation: <span id="demo"></span></p>
            <div class="slider-container">
                <button id="prevBtn">&lt;</button>
                <input type="range" min="0" max="0" class="slider" id="myRange">
                <button id="nextBtn">&gt;</button>
            </div>
            <iframe id="chartFrame"></iframe>
            <script>
                var filesByCurrency = {json.dumps(files_by_currency)};
                var currencySelect = document.getElementById("currencySelect");
                var slider = document.getElementById("myRange");
                var output = document.getElementById("demo");
                var iframe = document.getElementById("chartFrame");
                var prevBtn = document.getElementById("prevBtn");
                var nextBtn = document.getElementById("nextBtn");

                function updateFiles() {{
                    var currency = currencySelect.value;
                    var files = filesByCurrency[currency];
                    slider.max = files.length - 1;
                    slider.value = 0;
                    output.innerHTML = slider.value;
                    iframe.src = files[0];
                    slider.oninput = function() {{
                        output.innerHTML = this.value;
                        iframe.src = files[this.value];
                    }};
                    prevBtn.onclick = function() {{
                        if (slider.value > 0) {{
                            slider.value = parseInt(slider.value) - 1;
                            output.innerHTML = slider.value;
                            iframe.src = files[slider.value];
                        }}
                    }};
                    nextBtn.onclick = function() {{
                        if (slider.value < slider.max) {{
                            slider.value = parseInt(slider.value) + 1;
                            output.innerHTML = slider.value;
                            iframe.src = files[slider.value];
                        }}
                    }};
                }}

                currencySelect.onchange = updateFiles;
                updateFiles();
            </script>
        </body>
        </html>
        """
        
        # Save the master HTML file in the "viz" directory
        with open(self.master_chart_path, "w") as file:
            file.write(html_content)
        
        print("Master HTML file has been created in the viz_output directory.")

    def open_browser(self):
        """Open the web browser to the specified page."""
        web_page_url = f"http://localhost:{self.PORT}/master_chart.html"
        webbrowser.open_new(f"{web_page_url}#tab={self.window_name}")

    def close_existing_browser_tab(self):
        """Attempt to close the existing browser tab."""
        webbrowser.open_new(f"about:blank#tab={self.window_name}")

    def start_server(self):
        """Start the HTTP server."""
        os.chdir(self.dir_path)  # Change the working directory to the server directory
        
        # Check if the server is already running
        if self.server_thread is not None:
            print("Server is already running. Attempting to restart.")
            self.stop_server()  # Attempt to stop the server if it's running
        
        # Create a new server thread
        self.server_thread = Thread(target=self.run_server)
        self.server_thread.start()
        print(f"Serving at port {self.PORT}")
        # self.open_browser()  # Open the browser after the server starts

    def run_server(self):
        """Function to run the server in a thread."""
        with socketserver.TCPServer(("", self.PORT), self.Handler) as httpd:
            httpd.serve_forever()

    def stop_server(self):
        """Stop the HTTP server if it's running."""
        if self.server_thread is not None:
            print("Stopping the server.")
            self.server_thread = None
            # You may need to implement a proper shutdown mechanism for your server
            # For example, if using Flask: httpd.shutdown()


if __name__ == "__main__":
    viewer = Viewer()
