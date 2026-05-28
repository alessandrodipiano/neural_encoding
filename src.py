



import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_spikes(path):
    with open(path, "rb") as f:
        hdrchk = f.read(16)

        if b"DAN_SPK" in hdrchk:
            f.seek(828)
        else:
            f.seek(0)

        p = str(path)

        if ("mq" in p) or ("film02" in p) or ("film32" in p):
            dtype = "<u4"   # uint32, little-endian
        else:
            dtype = "<i4"   # int32, little-endian

        events = np.fromfile(f, dtype=dtype)

    events = events[events > 0]
    events_sec = events * 1e-4

    return events, events_sec


############################

#the following functions where taken from the matlba files created by the researcher 

###############################


HEADER_BYTES = 828


def load_log_lines(log_path):
    raw = Path(log_path).read_bytes()
    text = raw.replace(b"\x00", b" ").decode(errors="ignore")
    return [line.strip() for line in text.splitlines() if line.strip()]


def search_log(lines, section, key, default="JQK"):
    """
    Approximate Python equivalent of fsearch_log.
    Works for lines containing section/key/value text.
    """
    key_l = key.lower()
    section_l = section.lower()

    candidates = []
    for line in lines:
        low = line.lower()
        if key_l in low:
            candidates.append(line)

    if not candidates:
        return default

    line = candidates[0]

    # Try common patterns:
    # Key value
    # Key = value
    # Section Key value
    line = re.sub(r"\s+", " ", line)
    line = line.replace("=", " ")

    parts = line.split()
    for i, p in enumerate(parts):
        if p.lower() == key_l.lower() and i + 1 < len(parts):
            return " ".join(parts[i + 1:])

    # fallback: remove key name from line
    idx = line.lower().find(key_l)
    if idx >= 0:
        return line[idx + len(key):].strip(" :=\t")

    return default


def numbers(s, dtype=float):
    return np.array(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s)), dtype=dtype)


def read_sa0(path, refresh_rate=120, remote_refresh_rate=120):
    """
    Reads .sa0 spike file.

    Original MATLAB:
        data = fread(..., 'int32')
        data = data * refresh_rate / remote_refresh_rate
    These values are still in 1/10000 sec units after correction.
    """
    with open(path, "rb") as f:
        f.seek(HEADER_BYTES)
        data = np.fromfile(f, dtype="<i4")

    data = data.astype(float) * refresh_rate / remote_refresh_rate
    return data


def retrieve_log(path, filename, channels=None):
    """
    Python equivalent of fretrieve_log.m for tuning curve logs.
    """
    path = Path(path)
    log_path = path / filename
    lines = load_log_lines(log_path)

    flog = {
        "path": str(path),
        "log_file_name": filename,
    }

    file_type = search_log(lines, "FileInfo", "FileType", "JQK")
    file_version = search_log(lines, "FileInfo", "FileVersion", "JQK")
    test_type = search_log(lines, "TestInfo", "TestType", "JQK")

    flog["file_type"] = file_type
    flog["file_version"] = file_version
    flog["test_type"] = test_type
    flog["test_name"] = search_log(lines, "TestInfo", "Testname", filename)

    # The MATLAB checks these strictly. Here we keep parsing even if the text differs.
    remote_rr = numbers(search_log(lines, "TestInfo", "RemoteRefreshRate", "1"))
    rr = numbers(search_log(lines, "TestInfo", "RefreshRate", "1"))

    flog["remote_refresh_rate"] = float(remote_rr[0]) if len(remote_rr) else 1.0
    flog["refresh_rate"] = float(rr[0]) if len(rr) else flog["remote_refresh_rate"]

    tc_mode = search_log(lines, "TunningCurve", "TCMode", "JQK")
    old_style = tc_mode == "JQK"
    if old_style:
        tc_mode = "direction"

    flog["tc_mode"] = tc_mode
    flog["gratings_type"] = search_log(lines, "TunningCurve", "GratingsType", "JQK")

    def get_float(new_key, old_key=None, default="9999"):
        val = search_log(lines, "TunningCurve", new_key, default)
        if old_style and old_key is not None:
            val = search_log(lines, "TunningCurve", old_key, default)
        arr = numbers(val)
        return float(arr[0]) if len(arr) else float(default)

    def get_int(new_key, old_key=None, default="1"):
        val = search_log(lines, "TunningCurve", new_key, default)
        if old_style and old_key is not None:
            val = search_log(lines, "TunningCurve", old_key, default)
        arr = numbers(val, int)
        return int(arr[0]) if len(arr) else int(default)

    flog["ori_x"] = get_float("OriX")
    flog["ori_y"] = get_float("OriY")
    flog["gratings_w"] = get_float("GratingsW", "Width")
    flog["gratings_h"] = get_float("GratingsH", "Height")
    flog["gratings_spatial_frequency"] = get_float("GratingsSpatialFrequency", "SpatialFrequency")
    flog["gratings_velocity"] = get_float("GratingsVelocity", "Velocity")
    flog["bar_w"] = get_float("BarW")
    flog["bar_h"] = get_float("BarH")
    flog["inner_direction"] = get_float("InnerDirection")
    flog["outer_direction"] = get_float("OuterDirection")
    flog["circle_mask"] = search_log(lines, "TunningCurve", "CircleMask", "JQK")

    flog["single_test_time"] = get_int("SingleTestTime", "SingleTime", "1")
    flog["interval"] = get_int("Interval", default="0")
    flog["repeats"] = get_int("Repeats", default="1")
    flog["data_points"] = get_int("DataPoints", default="1")

    direction_test_mode = search_log(lines, "TunningCurve", "DirectionTestMode", "JQK")
    if old_style:
        direction_test_mode = "default"
    flog["direction_test_mode"] = direction_test_mode

    flog["direction_number"] = get_int("DirectionNumber", "Directions", "2")

    if old_style:
        flog["data_points"] = flog["direction_number"]

    # Lists
    direction_raw = search_log(lines, "TunningCurve", "DirectionList", "1 0")
    if old_style:
        direction_raw = search_log(lines, "TunningCurve", "DirectionSequence", "1 0")
    direction_nums = numbers(direction_raw)

    if old_style:
        flog["direction_list"] = direction_nums
    else:
        flog["direction_list"] = direction_nums[1:]

    spatial_nums = numbers(search_log(lines, "TunningCurve", "SpatialList", "1 0"))
    temporal_nums = numbers(search_log(lines, "TunningCurve", "TemporalList", "1 0"))
    test_nums = numbers(search_log(lines, "TunningCurve", "TestSequence", "1 0"), int)

    flog["spatial_list"] = spatial_nums[1:]
    flog["temporal_list"] = temporal_nums[1:]

    if old_style:
        flog["test_sequence"] = (flog["direction_list"] / (360 / flog["direction_number"])).astype(int)
    else:
        flog["test_sequence"] = test_nums[1:].astype(int)

    # Spike files
    if channels is None:
        channel_letters = [chr(ord("a") + i) for i in range(26)]
    else:
        channel_letters = [chr(ord("a") + c) if isinstance(c, int) else c for c in channels]

    spike_files = []
    for ch in channel_letters:
        channel_on = False
        for cluster in range(6):
            key = f"SpikeFile{ch}{cluster}"
            sf = search_log(lines, "DataFile", key, "JQK")
            if sf != "JQK":
                channel_on = True
                spike_files.append(sf.split()[0])
        if not channel_on:
            break

    flog["spike_files"] = spike_files

    # Load spikes
    spikes = []
    for sf in spike_files:
        spk = read_sa0(
            path / sf,
            refresh_rate=flog["refresh_rate"],
            remote_refresh_rate=flog["remote_refresh_rate"],
        )
        spikes.append(spk)
    flog["spikes"] = spikes

    nfiles = len(spike_files)
    data_points = flog["data_points"]
    repeats = flog["repeats"]

    sums = np.zeros((nfiles, data_points))
    individuals = np.zeros((nfiles, repeats, data_points))

    full_step = flog["single_test_time"] + flog["interval"]

    for i, spks in enumerate(spikes):
        for spike_raw in spks:
            spike_sec = spike_raw / 10000.0

            step_index = int(np.floor(spike_sec / full_step))
            spike_remain = spike_sec - step_index * full_step

            step_index_1based = step_index + 1

            if (
                spike_remain > flog["interval"] * 0.5
                and spike_remain < flog["interval"] * 0.5 + flog["single_test_time"]
            ):
                if step_index < len(flog["test_sequence"]):
                    cond = int(flog["test_sequence"][step_index])
                    sums[i, cond] += 1

                    repeat_index = int(np.ceil(step_index_1based / data_points)) - 1
                    if 0 <= repeat_index < repeats:
                        individuals[i, repeat_index, cond] += 1

    flog["sum"] = sums
    flog["individuals"] = individuals
    flog["stds"] = individuals.std(axis=1, ddof=1) if repeats > 1 else np.zeros_like(sums)
    flog["stderrs"] = flog["stds"] / np.sqrt(repeats)
    flog["isvalid"] = True

    return flog


def plot_tori(info, flog):
    dirlist = np.sort(flog["test_sequence"][:flog["direction_number"]] *
                      (360 / flog["direction_number"]))

    rsum = info["rsum"]
    rr = info["rr"]
    f = info["f"]
    Pmax = info["Pxx"][:, info["prefind"]]

    tf = info["temporal_freq"]
    numdirs = flog["direction_number"]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(info["spike_file"])

    # Polar plot
    ax1 = fig.add_subplot(2, 3, 1, projection="polar")
    theta = np.deg2rad(np.r_[dirlist, dirlist[0]])
    radius = np.r_[rsum, rsum[0]]
    ax1.plot(theta, radius, "-")
    ax1.set_title("Direction tuning")

    # Cartesian tuning curve
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(dirlist, rsum, "o-")
    ax2.set_xlabel("Direction / orientation (degrees)")
    ax2.set_ylabel("Spikes")
    ax2.grid(True)

    # Text
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.axis("off")
    cell_type = "complex" if info["relmod"] < 1 else "simple"

    txt = (
        f"{info['spike_file']}\n\n"
        f"Temporal freq = {info['temporal_freq']:.3f} Hz\n"
        f"Spatial freq = {info['spatial_freq']:.3f}\n"
        f"Velocity = {flog['gratings_velocity']:.3f}\n\n"
        f"Preferred dir = {info['prefdir']:.0f}°\n"
        f"Null dir = {info['nulldir']:.0f}°\n"
        f"Direction index = {info['dirind']:.3f}\n\n"
        f"Relative modulation = {info['relmod']:.3f}\n"
        f"Cell type = {cell_type}"
    )
    ax3.text(0, 1, txt, va="top")

    # PSD
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(f, Pmax)
    ax4.axvline(tf, linestyle="-.")
    ax4.axvline(2 * tf, linestyle="-.")
    ax4.set_xlim(0, 3 * tf)
    ax4.set_xlabel("Temporal frequency")
    ax4.set_ylabel("Magnitude")
    ax4.set_title(f"RM = {info['relmod']:.3f}")
    ax4.grid(True)

    # Spike raster-like binned plot
    ax5 = fig.add_subplot(2, 3, 5)
    for i in range(numdirs):
        temp = rr[:, i]
        ind = np.where(temp >= 1)[0]
        ax5.plot([0, len(rr)], [i + 1, i + 1], "-")
        ax5.plot(ind, temp[ind] + i, ".", markersize=4)

    single_time = flog["single_test_time"]
    frame_rate = flog["remote_refresh_rate"]

    if tf > 0:
        marks = int(np.floor(single_time / (1 / tf)))
        for i in range(1, marks + 1):
            x = i * (1 / tf) * frame_rate
            ax5.axvline(x, linestyle="-.")

    ax5.set_xlabel("Frame bin")
    ax5.set_ylabel("Direction")
    ax5.set_yticks(np.arange(1, numdirs + 1, 3))
    ax5.set_yticklabels(dirlist[::3].astype(int))
    ax5.grid(True)

    plt.tight_layout()
    plt.show()



def nextpow2(n):
    return int(np.ceil(np.log2(n)))


def tori(flog, cluster_index=0, plot=True):
    """

    cluster_index is Python-style: 0 means first spike file.
    MATLAB used 1 for the first file.
    """

    if not flog.get("isvalid", False):
        raise ValueError("Invalid flog.")

    if not (
        flog["test_type"] == "tuning curve"
        and flog["tc_mode"] == "direction"
        and flog["direction_test_mode"] == "default"
    ):
        raise ValueError("This only supports default direction tuning tests.")

    spk = flog["spikes"][cluster_index]

    spatial_frequency = flog["gratings_spatial_frequency"]
    velocity = flog["gratings_velocity"]
    frame_rate = flog["remote_refresh_rate"]
    frame_dt = 1 / frame_rate

    grat_temp_frequency = spatial_frequency * velocity
    numdirs = flog["direction_number"]

    dirlist = flog["test_sequence"] * (360 / flog["direction_number"])

    single_time = flog["single_test_time"]
    interval = flog["interval"]

    nframes = int(np.ceil(frame_rate * numdirs * (single_time + interval)))
    skip = int(round(interval / frame_dt))
    dur = int(round(single_time / frame_dt))

    # Bin spikes into frame bins.
    r = np.zeros(nframes)
    for s in spk:
        idx = int(np.ceil(s / (frame_dt * 10000))) - 1
        if 0 <= idx < nframes:
            r[idx] += 1

    rr = np.zeros((int(np.ceil(single_time / frame_dt)) + 1, numdirs))
    ss = np.zeros((int(round(interval / frame_dt)), numdirs))

    # First grating
    start = int(round(skip / 2))
    temp = r[start:start + dur + 1]
    rr[:len(temp), 0] = temp

    for i in range(1, numdirs):
        ind = int(round(skip / 2) + i * dur + i * skip)
        temp = r[ind:ind + dur + 1]
        rr[:len(temp), i] = temp

        sind = int(round(skip / 2) + i * dur)
        temp = r[sind + 1:sind + skip + 1]
        ss[:len(temp), i] = temp

    rsum = rr.sum(axis=0)
    ssum = ss.sum(axis=0)

    # Sort directions
    order = np.argsort(dirlist[:numdirs])
    dirlist_sorted = dirlist[:numdirs][order]
    rsum = rsum[order]
    rr = rr[:, order]
    ssum = ssum[order]
    ss = ss[:, order]

    # PSD
    n = rr.shape[0]
    nfft = 2 ** nextpow2(n)

    fftr = np.fft.fft(rr, n=nfft, axis=0)
    num_unique = int(np.ceil((nfft + 1) / 2))
    fftr = fftr[:num_unique, :]

    magr = 2 * np.abs(fftr)
    magr[0, :] /= 2

    if nfft % 2 == 0:
        magr[-1, :] /= 2

    magr = magr / nfft

    f = np.arange(num_unique) * 2 / nfft
    f = f * (frame_rate / 2)

    Pxx = magr

    maxind = int(np.argmax(rsum))
    Pmax = Pxx[:, maxind]

    indL = np.where(f < grat_temp_frequency)[0]
    indU = np.where(f > grat_temp_frequency)[0]

    if len(indL) == 0 or len(indU) == 0 or Pmax[0] == 0:
        relmodL = np.nan
        relmodU = np.nan
        relmod = np.nan
    else:
        relmodL = Pmax[indL[-1]] / Pmax[0]
        relmodU = Pmax[indU[0]] / Pmax[0]
        relmod = (relmodU + relmodL) / 2

    nulldir_idx = (maxind + len(dirlist_sorted) // 2) % len(dirlist_sorted)

    r_pref = rr[:, maxind].mean()
    r_null = rr[:, nulldir_idx].mean()

    dirind = (r_pref - r_null) / (r_pref + r_null) if (r_pref + r_null) != 0 else np.nan

    kernel = np.array([0.3, 0.3, 0.3])
    temp = np.convolve(rsum, kernel, mode="full")[1:-1]

    r_pref_s = temp[maxind] / rr.shape[0]
    r_null_s = temp[nulldir_idx] / rr.shape[0]

    dirind_smooth = (
        (r_pref_s - r_null_s) / (r_pref_s + r_null_s)
        if (r_pref_s + r_null_s) != 0
        else np.nan
    )

    info = {
        "test_name": flog["test_name"],
        "spike_file_index": cluster_index,
        "spike_file": flog["spike_files"][cluster_index],
        "temporal_freq": grat_temp_frequency,
        "spatial_freq": spatial_frequency,
        "rr": rr,
        "rsum": rsum,
        "ss": ss,
        "ssum": ssum,
        "Pxx": Pxx,
        "f": f,
        "relmodL": relmodL,
        "relmodU": relmodU,
        "relmod": relmod,
        "dirind": dirind,
        "dirind_smooth": dirind_smooth,
        "prefdir": dirlist_sorted[maxind],
        "nulldir": dirlist_sorted[nulldir_idx],
        "prefind": maxind,
        "nullind": nulldir_idx,
    }

    if plot:
        plot_tori(info, flog)

    return info