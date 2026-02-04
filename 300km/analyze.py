import json
from collections import defaultdict, Counter

"""
Analysis script for the 300km coastline + sea JSON file.
Provides comprehensive statistics on ROIs, targets, and data distribution.
"""

# === Paths ===
json_file = '/home/wangyu/CCR/CCR_CoastlineCloudRemoval/ROI_filter/300km/combined_coastline_and_sea_300km.json'
output_summary = '/home/wangyu/CCR/CCR_CoastlineCloudRemoval/ROI_filter/300km/coastline_sea_summary_300km.json'

print("="*80)
print("COASTLINE + SEA JSON ANALYSIS (300km Buffer)")
print("="*80)

# === Load JSON ===
print(f"\n[Step 1] Loading JSON file...")
with open(json_file, 'r') as f:
    data = json.load(f)
print(f"   Loaded {len(data)} entries")

# === Analysis ===
print(f"\n[Step 2] Analyzing data...")

roi_targets = defaultdict(set)  # Track unique targets per ROI
roi_samples = defaultdict(int)  # Track sample count per ROI
roi_s2_counts = defaultdict(int)  # Track total S2 images per ROI
roi_s1_counts = defaultdict(int)  # Track total S1 images per ROI

total_targets = 0
total_s2 = 0
total_s1 = 0
entries_without_target = []
s2_per_sample = []
s1_per_sample = []

for key, value in data.items():
    roi_id = value['roi'][0]  # e.g., "roi300488"
    
    # Count samples per ROI
    roi_samples[roi_id] += 1
    
    # Track unique targets per ROI
    targets = value.get('target', [])
    for target in targets:
        target_path = target[1] if len(target) > 1 else str(target)
        roi_targets[roi_id].add(target_path)
    
    total_targets += len(targets)
    
    if len(targets) == 0:
        entries_without_target.append(key)
    
    # Count S2 TOA images
    s2_count = len(value.get('s2_toa', []))
    roi_s2_counts[roi_id] += s2_count
    total_s2 += s2_count
    s2_per_sample.append(s2_count)
    
    # Count S1 images
    s1_count = len(value.get('s1', []))
    roi_s1_counts[roi_id] += s1_count
    total_s1 += s1_count
    s1_per_sample.append(s1_count)

# Convert target sets to counts
roi_unique_target_counts = {roi: len(targets) for roi, targets in roi_targets.items()}

# Get distribution of unique target counts
target_count_distribution = Counter(roi_unique_target_counts.values())

# === Print Results ===
print(f"\n{'='*80}")
print(f"SUMMARY STATISTICS")
print(f"{'='*80}")

print(f"\n📊 BASIC STATISTICS:")
print(f"   Total entries:                    {len(data)}")
print(f"   Total unique ROIs:                {len(roi_samples)}")
print(f"   Samples per ROI (avg):            {len(data) / len(roi_samples):.2f}")
print(f"   Samples per ROI (min):            {min(roi_samples.values())}")
print(f"   Samples per ROI (max):            {max(roi_samples.values())}")

print(f"\n🎯 TARGET STATISTICS:")
print(f"   Total target instances:           {total_targets}")
print(f"   Total unique target images:       {sum(len(targets) for targets in roi_targets.values())}")
print(f"   Entries with targets:             {len(data) - len(entries_without_target)} ({(len(data) - len(entries_without_target))/len(data)*100:.2f}%)")
print(f"   Entries without targets:          {len(entries_without_target)} ({len(entries_without_target)/len(data)*100:.2f}%)")
print(f"   Targets per entry (avg):          {total_targets / len(data):.2f}")

print(f"\n📈 UNIQUE TARGETS PER ROI:")
print(f"   ROIs with unique targets:         {len(roi_unique_target_counts)}")
print(f"   Unique targets per ROI (avg):     {sum(roi_unique_target_counts.values()) / len(roi_unique_target_counts):.2f}")
print(f"   Unique targets per ROI (min):     {min(roi_unique_target_counts.values())}")
print(f"   Unique targets per ROI (max):     {max(roi_unique_target_counts.values())}")

print(f"\n📊 UNIQUE TARGET COUNT DISTRIBUTION:")
print(f"   Unique Targets | Number of ROIs")
print(f"   " + "-"*35)
for count in sorted(target_count_distribution.keys()):
    print(f"   {count:14d} | {target_count_distribution[count]:14d}")

print(f"\n🛰️  S2 TOA IMAGE STATISTICS:")
print(f"   Total S2 TOA images:              {total_s2}")
print(f"   S2 TOA per sample (avg):          {sum(s2_per_sample) / len(s2_per_sample):.2f}")
print(f"   S2 TOA per sample (min):          {min(s2_per_sample)}")
print(f"   S2 TOA per sample (max):          {max(s2_per_sample)}")
print(f"   S2 TOA per ROI (avg):             {sum(roi_s2_counts.values()) / len(roi_s2_counts):.2f}")

print(f"\n📡 S1 IMAGE STATISTICS:")
print(f"   Total S1 images:                  {total_s1}")
print(f"   S1 per sample (avg):              {sum(s1_per_sample) / len(s1_per_sample):.2f}")
print(f"   S1 per sample (min):              {min(s1_per_sample)}")
print(f"   S1 per sample (max):              {max(s1_per_sample)}")
print(f"   Samples with S1 data:             {sum(1 for c in s1_per_sample if c > 0)} ({sum(1 for c in s1_per_sample if c > 0)/len(s1_per_sample)*100:.2f}%)")
print(f"   S1 per ROI (avg):                 {sum(roi_s1_counts.values()) / len(roi_s1_counts):.2f}")

# === TOP/BOTTOM ROIs ===
print(f"\n🏆 TOP 10 ROIs BY SAMPLE COUNT:")
top_samples = sorted(roi_samples.items(), key=lambda x: x[1], reverse=True)[:10]
for i, (roi_id, count) in enumerate(top_samples, 1):
    unique_targets = len(roi_targets[roi_id])
    print(f"   {i:2d}. {roi_id}: {count} samples, {unique_targets} unique targets")

print(f"\n🔝 TOP 10 ROIs BY UNIQUE TARGET COUNT:")
top_targets = sorted(roi_unique_target_counts.items(), key=lambda x: x[1], reverse=True)[:10]
for i, (roi_id, count) in enumerate(top_targets, 1):
    samples = roi_samples[roi_id]
    print(f"   {i:2d}. {roi_id}: {count} unique targets, {samples} samples")

print(f"\n📉 BOTTOM 10 ROIs BY UNIQUE TARGET COUNT:")
bottom_targets = sorted(roi_unique_target_counts.items(), key=lambda x: x[1])[:10]
for i, (roi_id, count) in enumerate(bottom_targets, 1):
    samples = roi_samples[roi_id]
    print(f"   {i:2d}. {roi_id}: {count} unique targets, {samples} samples")

# === WARNINGS ===
if entries_without_target:
    print(f"\n⚠️  WARNING: {len(entries_without_target)} ENTRIES WITHOUT TARGETS")
    print(f"   Sample entries (first 5):")
    for entry_key in entries_without_target[:5]:
        print(f"      - {entry_key}")
    if len(entries_without_target) > 5:
        print(f"      ... and {len(entries_without_target) - 5} more")

# === DATA QUALITY METRICS ===
print(f"\n✅ DATA QUALITY METRICS:")
avg_targets_per_roi = sum(roi_unique_target_counts.values()) / len(roi_unique_target_counts)
print(f"   Average unique targets per ROI:   {avg_targets_per_roi:.2f}")
print(f"   Coverage rate (entries/ROI):      {len(data) / len(roi_samples):.2f}")
print(f"   Target availability:              {(len(data) - len(entries_without_target))/len(data)*100:.2f}%")
print(f"   S1 data availability:             {sum(1 for c in s1_per_sample if c > 0)/len(s1_per_sample)*100:.2f}%")

# === Additional Analysis: Sample Distribution ===
print(f"\n📊 SAMPLE DISTRIBUTION ANALYSIS:")
sample_count_distribution = Counter(roi_samples.values())
print(f"   Samples/ROI | Number of ROIs")
print(f"   " + "-"*35)
for count in sorted(sample_count_distribution.keys())[:20]:  # Show first 20
    print(f"   {count:11d} | {sample_count_distribution[count]:14d}")
if len(sample_count_distribution) > 20:
    print(f"   ... and {len(sample_count_distribution) - 20} more unique sample counts")

# === S2/S1 Data Coverage per ROI ===
print(f"\n📈 IMAGE AVAILABILITY PER ROI:")
rois_with_s1 = sum(1 for roi, count in roi_s1_counts.items() if count > 0)
rois_with_s2 = sum(1 for roi, count in roi_s2_counts.items() if count > 0)
print(f"   ROIs with S2 TOA data:            {rois_with_s2} ({rois_with_s2/len(roi_samples)*100:.2f}%)")
print(f"   ROIs with S1 data:                {rois_with_s1} ({rois_with_s1/len(roi_samples)*100:.2f}%)")
print(f"   ROIs with both S1 and S2:         {sum(1 for roi in roi_samples.keys() if roi_s1_counts[roi] > 0 and roi_s2_counts[roi] > 0)}")

print(f"\n{'='*80}")
print(f"Analysis complete!")
print(f"{'='*80}\n")

# === Save summary to file ===
summary_output = {
    'buffer_distance_km': 300,
    'total_entries': len(data),
    'total_unique_rois': len(roi_samples),
    'total_unique_targets_across_all_rois': sum(len(targets) for targets in roi_targets.values()),
    'total_target_instances': total_targets,
    'avg_samples_per_roi': len(data) / len(roi_samples),
    'avg_unique_targets_per_roi': avg_targets_per_roi,
    'target_count_distribution': dict(target_count_distribution),
    'sample_count_distribution': dict(sample_count_distribution),
    'entries_without_targets': len(entries_without_target),
    'total_s2_images': total_s2,
    'total_s1_images': total_s1,
    's1_availability_percentage': sum(1 for c in s1_per_sample if c > 0)/len(s1_per_sample)*100,
    's2_availability_per_roi_percentage': rois_with_s2/len(roi_samples)*100,
    's1_availability_per_roi_percentage': rois_with_s1/len(roi_samples)*100,
    'top_10_rois_by_samples': [{'roi': roi, 'samples': count, 'unique_targets': len(roi_targets[roi])} 
                                for roi, count in top_samples],
    'top_10_rois_by_targets': [{'roi': roi, 'unique_targets': count, 'samples': roi_samples[roi]} 
                                for roi, count in top_targets]
}

with open(output_summary, 'w') as f:
    json.dump(summary_output, f, indent=2)

print(f"📄 Summary saved to: {output_summary}\n")
