# PowerShell script to automate the installation of dependencies and setup on Windows 11

# Function to check if a command exists
function Command-Exists {
    param (
        [string]$command
    )
    $commandPath = Get-Command $command -ErrorAction SilentlyContinue
    return $commandPath -ne $null
}

# 1. Install Miniconda if not already installed
if (-not (Command-Exists 'conda')) {
    Write-Output "Miniconda not found. Installing Miniconda..."
    $minicondaInstaller = "Miniconda3-latest-Windows-x86_64.exe"
    Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/$minicondaInstaller" -OutFile $minicondaInstaller
    Start-Process -Wait -FilePath .\$minicondaInstaller -ArgumentList "/InstallationType=JustMe", "/AddToPath=1", "/RegisterPython=1", "/S", "/D=$env:USERPROFILE\Miniconda3"
    Remove-Item .\$minicondaInstaller
    RefreshEnv
} else {
    Write-Output "Miniconda is already installed."
}

# 2. Ensure the Anthropic API key is set in the environment or storage
if (-not $env:ANTHROPIC_API_KEY) {
    Write-Output "Anthropic API key not found. Please set it in the environment or storage."
    exit 1
}

# 3. Set up a Conda environment and install Python dependencies
Write-Output "Setting up Conda environment and installing Python dependencies..."
conda create -n computer_use_ootb python=3.11 -y
conda activate computer_use_ootb
pip install -r dev-requirements.txt

# 4. Start the interface
Write-Output "Starting the interface..."
python app.py
