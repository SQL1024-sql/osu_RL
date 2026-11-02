# Copyright (c) osu_rl. Licensed under the MIT Licence.
param(
    [string]$OsuRepo = "..\ppy-osu"
)

$ErrorActionPreference = "Stop"

Write-Host "=== Integrating osu_rl into osu!lazer ($OsuRepo) ===" -ForegroundColor Cyan

# 1. Ensure target directories exist
$modsDir = Join-Path $OsuRepo "osu.Game.Rulesets.Osu\Mods"
$rlbotDir = Join-Path $modsDir "RLBot"
$replaysDir = Join-Path $OsuRepo "osu.Game.Rulesets.Osu\Replays"
$resourcesDir = Join-Path $OsuRepo "osu.Game.Rulesets.Osu\Resources"
$binDir = Join-Path $OsuRepo "osu.Desktop\bin\Debug\net10.0"

New-Item -ItemType Directory -Force -Path $rlbotDir | Out-Null
New-Item -ItemType Directory -Force -Path $resourcesDir | Out-Null
if (Test-Path $binDir) {
    New-Item -ItemType Directory -Force -Path $binDir | Out-Null
}

# 2. Copy source files
$scriptDir = $PSScriptRoot
Write-Host "Copying Mod source files..."
Copy-Item (Join-Path $scriptDir "OsuModRLBot.cs") (Join-Path $modsDir "OsuModRLBot.cs") -Force
Copy-Item (Join-Path $scriptDir "NeuralNetwork.cs") (Join-Path $rlbotDir "NeuralNetwork.cs") -Force
Copy-Item (Join-Path $scriptDir "OsuRLBotGenerator.cs") (Join-Path $replaysDir "OsuRLBotGenerator.cs") -Force
Copy-Item (Join-Path $scriptDir "model_weights.bin") (Join-Path $resourcesDir "model_weights.bin") -Force
if (Test-Path $binDir) {
    Copy-Item (Join-Path $scriptDir "model_weights.bin") (Join-Path $binDir "model_weights.bin") -Force
}

# 3. Patch OsuRuleset.cs to register OsuModRLBot
$rulesetPath = Join-Path $OsuRepo "osu.Game.Rulesets.Osu\OsuRuleset.cs"
$rulesetContent = Get-Content $rulesetPath -Raw

if ($rulesetContent -notmatch "OsuModRLBot") {
    Write-Host "Registering OsuModRLBot in OsuRuleset.cs..."
    $oldPattern = "new MultiMod(new OsuModAutoplay(), new OsuModCinema()),"
    $newPattern = "new MultiMod(new OsuModAutoplay(), new OsuModCinema()),`r`n                        new OsuModRLBot(),"
    if ($rulesetContent.Contains($oldPattern)) {
        $rulesetContent = $rulesetContent.Replace($oldPattern, $newPattern)
        Set-Content -Path $rulesetPath -Value $rulesetContent -NoNewline
        Write-Host "OsuRuleset.cs patched successfully." -ForegroundColor Green
    } else {
        Write-Warning "Could not find standard Autoplay registration in OsuRuleset.cs. Please check manually."
    }
} else {
    Write-Host "OsuModRLBot is already registered in OsuRuleset.cs." -ForegroundColor Green
}

# The weights are not embedded: the generator loads model_weights.bin from the game's output folder copied above.

Write-Host "=== Integration complete! ===" -ForegroundColor Cyan
Write-Host "To build and run osu!lazer with RL Bot:"
Write-Host "  & `"`$env:LOCALAPPDATA\Microsoft\dotnet10\dotnet.exe`" build `"$OsuRepo\osu.Desktop`""
Write-Host "  & `"$OsuRepo\osu.Desktop\bin\Debug\net10.0\osu!.exe`""
