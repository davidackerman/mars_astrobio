# Setting Up Your NASA API Key

Quick guide to configure your NASA API key for downloading Mars rover data.

## Step 1: Get Your API Key

If you don't have one yet:

1. Go to https://api.nasa.gov/
2. Fill out the form (takes 30 seconds)
3. You'll receive your API key instantly via email

The API key is **free** and allows 1,000 requests per hour.

## Step 2: Add Key to .env File

The repository includes a `.env` file where you can store your API key securely.

### Option A: Edit .env File Directly

Open [.env](.env) in your text editor and add your key:

```bash
# Before:
NASA_API_KEY=

# After:
NASA_API_KEY=your_actual_key_here_with_no_quotes
```

**Important**: Don't use quotes around the key!

### Option B: Use Command Line

```bash
# Replace YOUR_KEY with your actual key
echo "NASA_API_KEY=YOUR_KEY" > .env
```

### Option C: Use Your IDE

1. Open `.env` file in VS Code or your IDE
2. Replace the empty value with your key
3. Save the file

## Step 3: Verify Setup

Run the test script to verify everything works:

```bash
# Activate Pixi environment
pixi shell

# Run test
python test_download.py
```

If successful, you should see:
```
✅ Loaded API key from .env file
   API Key: AbCdEfGh...********
```

## Security Notes

✅ **Safe**: The `.env` file is in `.gitignore` and will **never** be committed to git
✅ **Safe**: Your API key stays on your local machine
✅ **Safe**: Example file `.env.example` doesn't contain your real key

❌ **Don't**: Commit your `.env` file to version control
❌ **Don't**: Share your API key publicly

## Troubleshooting

### "NASA_API_KEY not set!"

**Problem**: The script can't find your API key

**Solutions**:
1. Make sure you saved the `.env` file after editing
2. Check that there are no quotes around the key
3. Make sure you're in the project directory when running scripts
4. Try explicitly setting the environment variable:
   ```bash
   export NASA_API_KEY="your_key"
   python test_download.py
   ```

### "Invalid API key" Error

**Problem**: The API key is rejected by NASA

**Solutions**:
1. Double-check you copied the entire key (no spaces)
2. Make sure it's the key from https://api.nasa.gov/ (not another NASA site)
3. Check if the key was deactivated (generate a new one)

### Can't Find .env File

**Problem**: `.env` file doesn't exist

**Solution**:
```bash
# Copy the example file
cp .env.example .env

# Edit it with your key
nano .env
# or
code .env
```

## Alternative: Environment Variable

If you prefer not to use a `.env` file, you can set the environment variable directly:

```bash
# Temporary (this session only)
export NASA_API_KEY="your_key"

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export NASA_API_KEY="your_key"' >> ~/.bashrc
source ~/.bashrc
```

## File Overview

- **`.env`** - Your actual API key (gitignored, safe to edit)
- **`.env.example`** - Template file (committed to git, shows format)
- **`.gitignore`** - Ensures `.env` is never committed

## Next Steps

Once your API key is configured:

1. ✅ Run test: `python test_download.py`
2. 📥 Download data: `python scripts/download_data.py --instrument watson --sols 0-10`
3. 🚀 Start building your biosignature detector!

## Need Help?

Check:
- [GETTING_STARTED.md](GETTING_STARTED.md) - Full getting started guide
- [DATA_PIPELINE_README.md](DATA_PIPELINE_README.md) - Data pipeline documentation
- NASA API Docs: https://api.nasa.gov/

Your API key is the only setup needed - everything else is ready to go! 🎉
