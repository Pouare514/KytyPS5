#Requires -Version 5.1
<#
.SYNOPSIS
  Build + run TLOU Kyty phase validation, then triage log → verdict JSON.

.EXAMPLE
  .\_validate_phase.ps1 -Phase 59 -Seconds 180
  .\_validate_phase.ps1 -Phase 59 -SkipBuild -TriageOnly -LogPath _kyty_tlou_p59val.txt
#>
param(
    [Parameter(Mandatory)]
    [ValidateRange(54, 99)]
    [int]$Phase,

    [int]$Seconds = 180,

    [switch]$SkipBuild,

    [switch]$TriageOnly,

    [string]$LogPath
)

$ErrorActionPreference = 'Stop'
$RepoRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Get-Location).Path
}

function Resolve-PhaseLogPath {
    param([int]$PhaseNum, [string]$Explicit)
    if (-not [string]::IsNullOrWhiteSpace($Explicit)) {
        if ([System.IO.Path]::IsPathRooted($Explicit)) {
            return $Explicit
        }
        return (Join-Path $RepoRoot $Explicit)
    }
    return (Join-Path $RepoRoot ("_kyty_tlou_p{0}val.txt" -f $PhaseNum))
}

function Invoke-BuildPhase {
    param([int]$PhaseNum)

    if ($SkipBuild -or $TriageOnly) {
        return
    }

    $buildScriptPhase = Join-Path $RepoRoot ("_build_p{0}_inc.cmd" -f $PhaseNum)
    $buildScriptGeneric = Join-Path $RepoRoot '_build_p59_inc.cmd'
    $scriptToRun = $null
    if (Test-Path -LiteralPath $buildScriptPhase) {
        $scriptToRun = $buildScriptPhase
    } elseif (Test-Path -LiteralPath $buildScriptGeneric) {
        $scriptToRun = $buildScriptGeneric
    } else {
        throw "No build script found for phase $PhaseNum"
    }

    Write-Host "BUILD script=$scriptToRun"
    $proc = Start-Process -FilePath 'cmd.exe' `
        -ArgumentList @('/c', "`"$scriptToRun`"") `
        -WorkingDirectory $RepoRoot `
        -Wait -PassThru -NoNewWindow
    if ($null -eq $proc -or $proc.ExitCode -ne 0) {
        $code = if ($null -eq $proc) { -1 } else { $proc.ExitCode }
        throw "Build failed for phase $PhaseNum with exit code $code"
    }
    Write-Host "BUILD_EXIT=0"
}

function Clear-EnvVar {
    param([string]$Name)
    Remove-Item -Path "Env:$Name" -ErrorAction SilentlyContinue
}

function Invoke-RunPhase {
    param(
        [int]$PhaseNum,
        [int]$DurationSec,
        [string]$ResolvedLog
    )

    if ($TriageOnly) {
        return
    }

    $installDir = Join-Path $RepoRoot 'build\windows\install'
    $exe = Join-Path $installDir 'kyty_emulator.exe'
    if (-not (Test-Path -LiteralPath $exe)) {
        throw "kyty_emulator.exe not found: $exe"
    }

    $envMap = @{
        'KYTY_PHASE32_PENDING0'   = '1'
        'VK_LOADER_LAYERS_DISABLE' = '*'
        'KYTY_PHASE52_SEED_HEAD'  = '0'
        'KYTY_PHASE54_FAKE_JOB'   = '0'
        'KYTY_PHASE55_FAKE_QUEUE' = '0'
        'KYTY_PHASE56_FAKE_COUNT' = '0'
    }
    foreach ($k in $envMap.Keys) {
        Set-Item -Path "Env:$k" -Value $envMap[$k]
    }
    if ($PhaseNum -eq 66) {
        Set-Item -Path 'Env:KYTY_PHASE66_MENU_RECYCLE' -Value '1'
    } else {
        Clear-EnvVar 'KYTY_PHASE66_MENU_RECYCLE'
    }
    Clear-EnvVar 'KYTY_PHASE41_LIVE_KEEP1'
    Clear-EnvVar 'KYTY_PHASE41_RECALL_KEEP1'
    Clear-EnvVar 'KYTY_PHASE51_FAILFAST'
    Clear-EnvVar 'KYTY_PHASE51_BYPASS_FLIP'

    $outFile = Join-Path $RepoRoot ("_kyty_tlou_p{0}val_out.txt" -f $PhaseNum)
    $errFile = Join-Path $RepoRoot ("_kyty_tlou_p{0}val_err.txt" -f $PhaseNum)
    Remove-Item -LiteralPath $ResolvedLog, $outFile, $errFile -ErrorAction SilentlyContinue
    $fatal = Join-Path $installDir '_kyty_fatal.txt'
    Remove-Item -LiteralPath $fatal -ErrorAction SilentlyContinue

    $argList = @(
        '--game', 'D:\PS5\PPSA03396-app\eboot.bin',
        '--vulkan-validation', 'false',
        '--printf-direction', 'File',
        '--printf-output-file', $ResolvedLog
    )

    Write-Host "RUN phase=$PhaseNum seconds=$DurationSec log=$ResolvedLog"
    $proc = Start-Process -FilePath $exe `
        -ArgumentList $argList `
        -WorkingDirectory $installDir `
        -RedirectStandardOutput $outFile `
        -RedirectStandardError $errFile `
        -PassThru
    Write-Host ("PID={0}" -f $proc.Id)

    Start-Sleep -Seconds $DurationSec

    if (-not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Write-Host 'STOPPED exit=forced'
    } else {
        Write-Host ("EXITED code={0}" -f $proc.ExitCode)
    }
    Write-Host 'RUN_DONE'
}

function Get-LastMatchLine {
    param(
        [string[]]$Lines,
        [string]$Pattern
    )
    $hit = $Lines | Where-Object { $_ -match $Pattern } | Select-Object -Last 1
    if ($null -eq $hit) {
        return $null
    }
    return [string]$hit
}

function Invoke-TriagePhase {
    param(
        [int]$PhaseNum,
        [string]$ResolvedLog
    )

    if (-not (Test-Path -LiteralPath $ResolvedLog)) {
        Write-Host "Log file not found: $ResolvedLog"
        exit 3
    }

    $errFile = Join-Path $RepoRoot ("_kyty_tlou_p{0}val_err.txt" -f $PhaseNum)
    $lines = @(Get-Content -LiteralPath $ResolvedLog -ErrorAction Stop)
    if (Test-Path -LiteralPath $errFile) {
        $lines = $lines + @(Get-Content -LiteralPath $errFile -ErrorAction SilentlyContinue)
    }

    $heatmapPat = "phase${PhaseNum} heatmap"
    $seedPat = "phase${PhaseNum} seed"
    $heatmapLine = Get-LastMatchLine -Lines $lines -Pattern $heatmapPat
    # Prefer the richest heatmap line (includes pre_agc / post_guest_real when present).
    if ($PhaseNum -eq 61) {
        $richHeat = Get-LastMatchLine -Lines $lines -Pattern 'phase61 heatmap .*pre_agc='
        if ($richHeat) {
            $heatmapLine = $richHeat
        }
    }
    if ($PhaseNum -eq 62) {
        $richHeat = Get-LastMatchLine -Lines $lines -Pattern 'phase62 heatmap .*res_used='
        if (-not $richHeat) {
            $richHeat = Get-LastMatchLine -Lines $lines -Pattern 'phase62 heatmap .*pre_agc='
        }
        if ($richHeat) {
            $heatmapLine = $richHeat
        }
        if (-not $heatmapLine) {
            $heatmapLine = Get-LastMatchLine -Lines $lines -Pattern 'phase62 unreg_gate'
        }
    }
    if ($PhaseNum -eq 63) {
        $richHeat = Get-LastMatchLine -Lines $lines -Pattern 'phase63 heatmap .*unreg_owners='
        if ($richHeat) {
            $heatmapLine = $richHeat
        }
    }
    if ($PhaseNum -eq 64) {
        $richHeat = Get-LastMatchLine -Lines $lines -Pattern 'phase64 heatmap .*main_cond_wait_post='
        if ($richHeat) {
            $heatmapLine = $richHeat
        }
    }
    if ($PhaseNum -eq 65) {
        $richHeat = Get-LastMatchLine -Lines $lines -Pattern 'phase65 heatmap .*wait_alias_main='
        if (-not $richHeat) {
            $richHeat = Get-LastMatchLine -Lines $lines -Pattern 'phase65 heatmap .*wait_main='
        }
        if ($richHeat) {
            $heatmapLine = $richHeat
        }
    }
    if ($PhaseNum -eq 66) {
        $richHeat = Get-LastMatchLine -Lines $lines -Pattern 'phase66 heatmap .*recycle_n='
        if ($richHeat) {
            $heatmapLine = $richHeat
        }
    }
    $seedLine = Get-LastMatchLine -Lines $lines -Pattern $seedPat
    $submitLine = Get-LastMatchLine -Lines $lines -Pattern 'phase\d+ submit kind='
    if (-not $submitLine) {
        $submitLine = Get-LastMatchLine -Lines $lines -Pattern 'submit kind='
    }
    $guestReal = ($lines | Where-Object {
            $_ -match 'guest_real=\d|kind=guest_real|submit_guest_real=' -and
            $_ -notmatch 'heatmap'
        } | Select-Object -Last 1)
    if ($null -ne $guestReal) {
        $guestReal = [string]$guestReal
    }
    $fatalLine = Get-LastMatchLine -Lines $lines -Pattern 'Not implemented|FATAL|g_pthread_self'

    $cause = $null
    if ($heatmapLine -and ($heatmapLine -match 'cause=(\S+)')) {
        $cause = $Matches[1]
    } elseif ($null -ne (Get-LastMatchLine -Lines $lines -Pattern 'cause=(\S+)')) {
        $causeLine = Get-LastMatchLine -Lines $lines -Pattern 'cause='
        if ($causeLine -match 'cause=(\S+)') {
            $cause = $Matches[1]
        }
    }

    $crash = [bool]$fatalLine
    $exitCode = 0
    if (-not $heatmapLine) {
        Write-Host 'No heatmap line found in log.'
        $exitCode = 3
    } elseif ($crash) {
        $exitCode = 2
    }

    $verdict = [ordered]@{
        Phase         = $PhaseNum
        Cause         = $cause
        Heatmap       = $heatmapLine
        Seed          = $seedLine
        SubmitLine    = $submitLine
        GuestRealLine = $guestReal
        FatalLine     = $fatalLine
        Crash         = $crash
        Timestamp     = (Get-Date).ToString('o')
        LogFile       = $ResolvedLog
    }

    if ($PhaseNum -eq 61) {
        $ringPm4Total = $null
        $ringPtrTotal = $null
        $ringMut = $null
        $ringRich = $null
        $preAgc = $null
        $postGuestReal = $null
        if ($heatmapLine) {
            if ($heatmapLine -match 'pm4_total=(\d+)') { $ringPm4Total = [int]$Matches[1] }
            if ($heatmapLine -match 'ptr_total=(\d+)') { $ringPtrTotal = [int]$Matches[1] }
            if ($heatmapLine -match 'mut=(\d+)') { $ringMut = [int]$Matches[1] }
            if ($heatmapLine -match 'rich=(\d+)') { $ringRich = [int]$Matches[1] }
            if ($heatmapLine -match 'pre_agc=(\d+)') { $preAgc = [int]$Matches[1] }
            if ($heatmapLine -match 'post_guest_real=(\d+)') { $postGuestReal = [int]$Matches[1] }
        }
        $probeLine = Get-LastMatchLine -Lines $lines -Pattern 'phase61 ring_probe '
        if ($null -eq $ringPm4Total -and $probeLine -and ($probeLine -match 'pm4=(\d+)')) {
            $ringPm4Total = [int]$Matches[1]
        }
        $verdict['RingPm4Total'] = $ringPm4Total
        $verdict['RingPtrTotal'] = $ringPtrTotal
        $verdict['RingMutatingProbes'] = $ringMut
        $verdict['RingRichProbes'] = $ringRich
        $verdict['PreUnregAgcCalls'] = $preAgc
        $verdict['PostUnregGuestRealSubmits'] = $postGuestReal
        $verdict['LastRingProbe'] = $probeLine
    }

    if ($PhaseNum -eq 62) {
        $preAgc = $null
        $postGuestReal = $null
        $postSeed = $null
        $nonKpriMut = $null
        $alt = $null
        $unregTsc = $null
        $resUsed = $null
        if ($heatmapLine) {
            if ($heatmapLine -match 'pre_agc=(\d+)') { $preAgc = [int]$Matches[1] }
            if ($heatmapLine -match 'post_guest_real=(\d+)') { $postGuestReal = [int]$Matches[1] }
            if ($heatmapLine -match 'post_seed=(\d+)') { $postSeed = [int]$Matches[1] }
            if ($heatmapLine -match 'non_kpri_mut=(\d+)') { $nonKpriMut = [int]$Matches[1] }
            if ($heatmapLine -match 'alt=(\d+)') { $alt = [int]$Matches[1] }
            if ($heatmapLine -match 'unreg_tsc=(\d+)') { $unregTsc = [long]$Matches[1] }
            if ($heatmapLine -match 'res_used=(\d+)') { $resUsed = [int]$Matches[1] }
        }
        $unregLine = Get-LastMatchLine -Lines $lines -Pattern 'phase62 unreg_gate'
        $resLine = Get-LastMatchLine -Lines $lines -Pattern 'phase62 resource '
        $verdict['PreAgc'] = $preAgc
        $verdict['PostGuestReal'] = $postGuestReal
        $verdict['PostSeed'] = $postSeed
        $verdict['NonKpriMut'] = $nonKpriMut
        $verdict['AltAnchor'] = $alt
        $verdict['UnregTsc'] = $unregTsc
        $verdict['ResUsed'] = $resUsed
        $verdict['UnregGate'] = $unregLine
        $verdict['LastResource'] = $resLine
    }

    if ($PhaseNum -eq 63) {
        $unregOwners = $null
        $unregRes = $null
        $submitAttempt = $null
        $submitGuestReal = $null
        $postAgc = $null
        if ($heatmapLine) {
            if ($heatmapLine -match 'unreg_owners=(\d+)') { $unregOwners = [int]$Matches[1] }
            if ($heatmapLine -match 'unreg_res_total=(\d+)') { $unregRes = [int]$Matches[1] }
            if ($heatmapLine -match 'submit_attempt_post=(\d+)') { $submitAttempt = [int]$Matches[1] }
            if ($heatmapLine -match 'submit_guest_real=(\d+)') { $submitGuestReal = [int]$Matches[1] }
            if ($heatmapLine -match 'post_agc=(\d+)') { $postAgc = [int]$Matches[1] }
        }
        $entryLine = Get-LastMatchLine -Lines $lines -Pattern 'phase63 submit_entry '
        $unregLine = Get-LastMatchLine -Lines $lines -Pattern 'phase63 unregister '
        $verdict['UnregisterOwners'] = $unregOwners
        $verdict['UnregisterResourcesTotal'] = $unregRes
        $verdict['SubmitAttemptsPost'] = $submitAttempt
        $verdict['SubmitGuestRealPost'] = $submitGuestReal
        $verdict['PostAgcCalls'] = $postAgc
        $verdict['LastSubmitEntry'] = $entryLine
        $verdict['LastUnregister'] = $unregLine
    }

    if ($PhaseNum -eq 64) {
        $mainWait = $null
        $mainSig = $null
        $flipStuck = $null
        $ndjobStatic = $null
        $guestReal = $null
        if ($heatmapLine) {
            if ($heatmapLine -match 'main_cond_wait_post=(\d+)') { $mainWait = [int]$Matches[1] }
            if ($heatmapLine -match 'main_cond_signal_post=(\d+)') { $mainSig = [int]$Matches[1] }
            if ($heatmapLine -match 'flip_pending_stuck=(\d+)') { $flipStuck = [int]$Matches[1] }
            if ($heatmapLine -match 'ndjob_static=(\d+)') { $ndjobStatic = [int]$Matches[1] }
            if ($heatmapLine -match 'guest_real_post=(\d+)') { $guestReal = [int]$Matches[1] }
        }
        $verdict['MainCondWaitPost'] = $mainWait
        $verdict['MainCondSignalPost'] = $mainSig
        $verdict['FlipPendingStuck'] = $flipStuck
        $verdict['NdjobStatic'] = $ndjobStatic
        $verdict['GuestRealPost'] = $guestReal
    }

    if ($PhaseNum -eq 65) {
        $waitMain = $null
        $waitMixed = $null
        $waitCompute = $null
        $waitOther = $null
        $waitAlias = $null
        $mainAlive = $null
        $guestReg2 = $null
        $guestFlip = $null
        $guestReal = $null
        $hostFlip = $null
        if ($heatmapLine) {
            if ($heatmapLine -match 'wait_main=(\d+)') { $waitMain = [int]$Matches[1] }
            if ($heatmapLine -match 'wait_mixed=(\d+)') { $waitMixed = [int]$Matches[1] }
            if ($heatmapLine -match 'wait_compute=(\d+)') { $waitCompute = [int]$Matches[1] }
            if ($heatmapLine -match 'wait_other=(\d+)') { $waitOther = [int]$Matches[1] }
            if ($heatmapLine -match 'wait_alias_main=(\d+)') { $waitAlias = [int]$Matches[1] }
            if ($heatmapLine -match 'main_alive=(\d+)') { $mainAlive = [int]$Matches[1] }
            if ($heatmapLine -match 'guest_regbuf2=(\d+)') { $guestReg2 = [int]$Matches[1] }
            if ($heatmapLine -match 'guest_flip_seen=(\d+)') { $guestFlip = [int]$Matches[1] }
            if ($heatmapLine -match 'guest_real_post=(\d+)') { $guestReal = [int]$Matches[1] }
            if ($heatmapLine -match 'host_flip_active=(\d+)') { $hostFlip = [int]$Matches[1] }
        }
        $verdict['WaitMain'] = $waitMain
        $verdict['WaitMixed'] = $waitMixed
        $verdict['WaitCompute'] = $waitCompute
        $verdict['WaitOther'] = $waitOther
        $verdict['WaitAliasMain'] = $waitAlias
        $verdict['MainAlive'] = $mainAlive
        $verdict['GuestRegbuf2'] = $guestReg2
        $verdict['GuestFlipSeen'] = $guestFlip
        $verdict['GuestRealPost'] = $guestReal
        $verdict['HostFlipActive'] = $hostFlip
    }

    if ($PhaseNum -eq 66) {
        $recycleN = $null
        $pendingStreak = $null
        $guestReg2 = $null
        $guestFlip = $null
        $hostFlip = $null
        if ($heatmapLine) {
            if ($heatmapLine -match 'recycle_n=(\d+)') { $recycleN = [int]$Matches[1] }
            if ($heatmapLine -match 'pending_streak=(\d+)') { $pendingStreak = [int]$Matches[1] }
            if ($heatmapLine -match 'guest_regbuf2=(\d+)') { $guestReg2 = [int]$Matches[1] }
            if ($heatmapLine -match 'guest_flip_seen=(\d+)') { $guestFlip = [int]$Matches[1] }
            if ($heatmapLine -match 'host_flip_active=(\d+)') { $hostFlip = [int]$Matches[1] }
        }
        $recycleLine = Get-LastMatchLine -Lines $lines -Pattern 'phase66 menu_recycle flip'
        $verdict['RecycleN'] = $recycleN
        $verdict['PendingStreak'] = $pendingStreak
        $verdict['GuestRegbuf2'] = $guestReg2
        $verdict['GuestFlipSeen'] = $guestFlip
        $verdict['HostFlipActive'] = $hostFlip
        $verdict['LastRecycle'] = $recycleLine
    }

    $jsonFile = Join-Path $RepoRoot ("_kyty_tlou_p{0}_verdict.json" -f $PhaseNum)
    ($verdict | ConvertTo-Json -Depth 4) | Set-Content -LiteralPath $jsonFile -Encoding UTF8

    Write-Host ("Phase {0} verdict:" -f $PhaseNum)
    Write-Host ("  CAUSE      = {0}" -f $verdict.Cause)
    Write-Host ("  CRASH      = {0}" -f $verdict.Crash)
    Write-Host ("  HEATMAP    = {0}" -f $verdict.Heatmap)
    Write-Host ("  SEED       = {0}" -f $verdict.Seed)
    Write-Host ("  SUBMIT     = {0}" -f $verdict.SubmitLine)
    Write-Host ("  GUEST_REAL = {0}" -f $verdict.GuestRealLine)
    if ($PhaseNum -eq 61) {
        Write-Host ("  RING_PM4   = {0}" -f $verdict.RingPm4Total)
        Write-Host ("  RING_PTR   = {0}" -f $verdict.RingPtrTotal)
        Write-Host ("  RING_MUT   = {0}" -f $verdict.RingMutatingProbes)
        Write-Host ("  RING_RICH  = {0}" -f $verdict.RingRichProbes)
        Write-Host ("  PRE_AGC    = {0}" -f $verdict.PreUnregAgcCalls)
        Write-Host ("  POST_GR    = {0}" -f $verdict.PostUnregGuestRealSubmits)
    }
    if ($PhaseNum -eq 62) {
        Write-Host ("  PRE_AGC    = {0}" -f $verdict.PreAgc)
        Write-Host ("  POST_GR    = {0}" -f $verdict.PostGuestReal)
        Write-Host ("  POST_SEED  = {0}" -f $verdict.PostSeed)
        Write-Host ("  NON_KPRI_M = {0}" -f $verdict.NonKpriMut)
        Write-Host ("  ALT        = {0}" -f $verdict.AltAnchor)
        Write-Host ("  UNREG_TSC  = {0}" -f $verdict.UnregTsc)
        Write-Host ("  RES_USED   = {0}" -f $verdict.ResUsed)
    }
    if ($PhaseNum -eq 63) {
        Write-Host ("  UNREG_OWN  = {0}" -f $verdict.UnregisterOwners)
        Write-Host ("  UNREG_RES  = {0}" -f $verdict.UnregisterResourcesTotal)
        Write-Host ("  SUBMIT_ATT = {0}" -f $verdict.SubmitAttemptsPost)
        Write-Host ("  SUBMIT_GR  = {0}" -f $verdict.SubmitGuestRealPost)
        Write-Host ("  POST_AGC   = {0}" -f $verdict.PostAgcCalls)
    }
    if ($PhaseNum -eq 64) {
        Write-Host ("  MAIN_WAIT  = {0}" -f $verdict.MainCondWaitPost)
        Write-Host ("  MAIN_SIG   = {0}" -f $verdict.MainCondSignalPost)
        Write-Host ("  FLIP_STUCK = {0}" -f $verdict.FlipPendingStuck)
        Write-Host ("  NDJOB_STAT = {0}" -f $verdict.NdjobStatic)
        Write-Host ("  GUEST_REAL = {0}" -f $verdict.GuestRealPost)
    }
    if ($PhaseNum -eq 65) {
        Write-Host ("  WAIT_MAIN  = {0}" -f $verdict.WaitMain)
        Write-Host ("  WAIT_MIXED = {0}" -f $verdict.WaitMixed)
        Write-Host ("  WAIT_COMP  = {0}" -f $verdict.WaitCompute)
        Write-Host ("  WAIT_OTHER = {0}" -f $verdict.WaitOther)
        Write-Host ("  WAIT_ALIAS = {0}" -f $verdict.WaitAliasMain)
        Write-Host ("  MAIN_ALIVE = {0}" -f $verdict.MainAlive)
        Write-Host ("  GUEST_REG2 = {0}" -f $verdict.GuestRegbuf2)
        Write-Host ("  GUEST_FLIP = {0}" -f $verdict.GuestFlipSeen)
        Write-Host ("  GUEST_REAL = {0}" -f $verdict.GuestRealPost)
        Write-Host ("  HOST_FLIP  = {0}" -f $verdict.HostFlipActive)
    }
    if ($PhaseNum -eq 66) {
        Write-Host ("  RECYCLE_N  = {0}" -f $verdict.RecycleN)
        Write-Host ("  PEND_STREAK= {0}" -f $verdict.PendingStreak)
        Write-Host ("  GUEST_REG2 = {0}" -f $verdict.GuestRegbuf2)
        Write-Host ("  GUEST_FLIP = {0}" -f $verdict.GuestFlipSeen)
        Write-Host ("  HOST_FLIP  = {0}" -f $verdict.HostFlipActive)
        Write-Host ("  LAST_RECYC = {0}" -f $verdict.LastRecycle)
    }
    Write-Host ("  JSON       = {0}" -f $jsonFile)

    exit $exitCode
}

# --- main ---
$resolvedLog = Resolve-PhaseLogPath -PhaseNum $Phase -Explicit $LogPath

try {
    Invoke-BuildPhase -PhaseNum $Phase
    Invoke-RunPhase -PhaseNum $Phase -DurationSec $Seconds -ResolvedLog $resolvedLog
    Invoke-TriagePhase -PhaseNum $Phase -ResolvedLog $resolvedLog
} catch {
    Write-Host ("Error: {0}" -f $_.Exception.Message)
    exit 2
}
