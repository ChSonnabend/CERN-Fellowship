#include "SimulationDataFormat/O2DatabasePDG.h"

#include <TChain.h>
#include <TDatabasePDG.h>
#include <TFile.h>
#include <TParticlePDG.h>
#include <TSystem.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{
constexpr uint64_t NBitsTrackID = 31;
constexpr uint64_t NBitsEventID = 19;
constexpr uint64_t NBitsSourceID = 8;
constexpr uint64_t MaskTrackID = (uint64_t{1} << NBitsTrackID) - 1;
constexpr uint64_t MaskEventID = (uint64_t{1} << NBitsEventID) - 1;
constexpr uint64_t MaskSourceID = (uint64_t{1} << NBitsSourceID) - 1;

std::vector<std::string> getInputFiles(const std::string& input)
{
  std::vector<std::string> files;
  std::filesystem::path path(input);
  if (std::filesystem::is_regular_file(path)) {
    files.emplace_back(path.string());
    return files;
  }
  if (std::filesystem::is_directory(path)) {
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
      const auto name = entry.path().filename().string();
      constexpr std::string_view suffix = "_Kine.root";
      if (entry.is_regular_file() && name.size() >= suffix.size() &&
          name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
        files.emplace_back(entry.path().string());
      }
    }
    std::sort(files.begin(), files.end());
  }
  return files;
}

uint64_t mcLabelRaw(int trackID, int eventID, int sourceID)
{
  return (static_cast<uint64_t>(trackID) & MaskTrackID) |
         ((static_cast<uint64_t>(eventID) & MaskEventID) << NBitsTrackID) |
         ((static_cast<uint64_t>(sourceID) & MaskSourceID) << (NBitsTrackID + NBitsEventID));
}

int chargeSign(int pdg)
{
  if (pdg == 0) {
    return 0;
  }
  if (std::abs(pdg) >= 1000000000) {
    const auto z = (std::abs(pdg) / 10000) % 1000;
    return z == 0 ? 0 : (pdg > 0 ? 1 : -1);
  }
  const auto particle = TDatabasePDG::Instance()->GetParticle(pdg);
  if (!particle) {
    return 0;
  }
  const auto charge = particle->Charge() / 3.0;
  return charge > 0.0 ? 1 : (charge < 0.0 ? -1 : 0);
}

double massFromPDG(int pdg)
{
  bool success = false;
  const auto mass = o2::O2DatabasePDG::Mass(pdg, success);
  if (success) {
    return mass;
  }
  const auto particle = TDatabasePDG::Instance()->GetParticle(pdg);
  return particle ? particle->Mass() : 0.0;
}

double safeEta(double p, double pz)
{
  return p > std::abs(pz) ? 0.5 * std::log((p + pz) / (p - pz)) : std::numeric_limits<double>::quiet_NaN();
}

double safeRapidity(double energy, double pz)
{
  return (energy > std::abs(pz)) ? 0.5 * std::log((energy + pz) / (energy - pz)) : std::numeric_limits<double>::quiet_NaN();
}

int popcount22(uint32_t hitMask)
{
  return __builtin_popcount(hitMask & ((uint32_t{1} << 22) - 1));
}

std::string fileBasename(const char* name)
{
  return std::filesystem::path(name ? name : "").filename().string();
}

int clippedBin(double value, double min, double max, int nBins)
{
  if (nBins <= 1 || !(max > min) || !std::isfinite(value)) {
    return 0;
  }
  if (value <= min) {
    return 0;
  }
  if (value >= max) {
    return nBins - 1;
  }
  return std::clamp(static_cast<int>((value - min) / (max - min) * nBins), 0, nBins - 1);
}

int balanceBinIndex(int phiBin, int radiusBin, int etaBin, int labelBin, int nPhiBins, int nRadiusBins, int nEtaBins)
{
  return (((phiBin * nRadiusBins + radiusBin) * nEtaBins + etaBin) * 2) + labelBin;
}
} // namespace

void extract_o2_kine_training(
  const char* input = "/lustre/alice/users/csonnab/PhD/jobs/simulation/sim_data/training/o2sim_09092026_anchoredMC_24arp2_559843/SC/0_100/500EV_1/tf1",
  const char* output = "/lustre/alice/users/csonnab/cern-fellowship/run/simnet/data/output/o2_kine_training.csv",
  long long maxEvents = -1,
  long long maxTracks = -1,
  long long progressEvery = 100000,
  bool downsample = false,
  int nPhiBins = 36,
  int nRadiusBins = 30,
  int nEtaBins = 20,
  double radiusMax = 500.0,
  double etaMin = -10.0,
  double etaMax = 10.0)
{
  const auto files = getInputFiles(input);
  if (files.empty()) {
    std::cerr << "No *_Kine.root files found for input: " << input << "\n";
    return;
  }

  std::filesystem::create_directories(std::filesystem::path(output).parent_path());

  TChain chain("o2sim");
  for (const auto& file : files) {
    chain.Add(file.c_str());
  }

  TTreeReader reader(&chain);
  TTreeReaderArray<float> px(reader, "MCTrack.mStartVertexMomentumX");
  TTreeReaderArray<float> py(reader, "MCTrack.mStartVertexMomentumY");
  TTreeReaderArray<float> pz(reader, "MCTrack.mStartVertexMomentumZ");
  TTreeReaderArray<float> vx(reader, "MCTrack.mStartVertexCoordinatesX");
  TTreeReaderArray<float> vy(reader, "MCTrack.mStartVertexCoordinatesY");
  TTreeReaderArray<float> vz(reader, "MCTrack.mStartVertexCoordinatesZ");
  TTreeReaderArray<float> tns(reader, "MCTrack.mStartVertexCoordinatesT");
  TTreeReaderArray<float> weight(reader, "MCTrack.mWeight");
  TTreeReaderArray<int> pdg(reader, "MCTrack.mPdgCode");
  TTreeReaderArray<int> mother(reader, "MCTrack.mMotherTrackId");
  TTreeReaderArray<int> secondMother(reader, "MCTrack.mSecondMotherTrackId");
  TTreeReaderArray<int> firstDaughter(reader, "MCTrack.mFirstDaughterTrackId");
  TTreeReaderArray<int> lastDaughter(reader, "MCTrack.mLastDaughterTrackId");
  TTreeReaderArray<int> propRaw(reader, "MCTrack.mProp");
  TTreeReaderArray<int> statusCode(reader, "MCTrack.mStatusCode");
  TTreeReaderValue<double> eventVx(reader, "MCEventHeader.FairMCEventHeader.fX");
  TTreeReaderValue<double> eventVy(reader, "MCEventHeader.FairMCEventHeader.fY");
  TTreeReaderValue<double> eventVz(reader, "MCEventHeader.FairMCEventHeader.fZ");

  const long long nBalanceBins = static_cast<long long>(nPhiBins) * nRadiusBins * nEtaBins * 2;
  long long quotaPerBin = std::numeric_limits<long long>::max();
  std::vector<long long> binCounts;
  if (downsample) {
    if (maxTracks <= 0) {
      std::cerr << "Balanced downsampling requires maxTracks > 0. Continuing without downsampling.\n";
      downsample = false;
    } else if (nPhiBins <= 0 || nRadiusBins <= 0 || nEtaBins <= 0 || radiusMax <= 0.0 || etaMax <= etaMin) {
      std::cerr << "Invalid downsampling bin configuration. Continuing without downsampling.\n";
      downsample = false;
    } else {
      quotaPerBin = std::max<long long>(1, maxTracks / nBalanceBins);
      binCounts.assign(nBalanceBins, 0);
      std::cerr << "Balanced downsampling enabled: " << nPhiBins << " phi bins x " << nRadiusBins
                << " radius bins x " << nEtaBins << " eta bins x 2 labels = " << nBalanceBins
                << " bins, quota " << quotaPerBin << " tracks/bin"
                << " (radius range [0," << radiusMax << "] cm, eta range [" << etaMin << "," << etaMax << "])\n";
    }
  }

  std::ofstream out(output);
  out << std::setprecision(9);
  out << "source_id,event_id,track_id,mc_label_raw,input_file,pdg,abs_pdg,charge_sign,mass,energy,ekin,"
      << "px,py,pz,p,pt,eta,phi,theta,rapidity,vx,vy,vz,t_ns,event_vx,event_vy,event_vz,"
      << "dx_from_event,dy_from_event,dz_from_event,r_xy,r_from_event_xy,r3_from_event,"
      << "mother_id,second_mother_id,first_daughter_id,last_daughter_id,direct_daughter_count,has_daughters,"
      << "process,status_code,weight,hit_mask,num_detectors_with_hits,has_hits,to_be_done,inhibited,"
      << "is_transported,is_primary,kept_by_o2,can_avoid_geant\n";

  long long totalRows = 0;
  long long entry = 0;
  int lastTreeNumber = -1;
  int sourceID = 0;
  std::string currentFile;

  while (reader.Next()) {
    if (maxEvents >= 0 && entry >= maxEvents) {
      break;
    }
    if (chain.GetTreeNumber() != lastTreeNumber) {
      lastTreeNumber = chain.GetTreeNumber();
      sourceID = lastTreeNumber;
      currentFile = fileBasename(chain.GetFile() ? chain.GetFile()->GetName() : "");
    }

    const int nTracks = px.GetSize();
    std::vector<int> daughterCounts(nTracks, 0);
    for (int trackID = 0; trackID < nTracks; ++trackID) {
      const int mid = mother[trackID];
      if (mid >= 0 && mid < nTracks) {
        ++daughterCounts[mid];
      }
    }

    for (int trackID = 0; trackID < nTracks; ++trackID) {
      const double x = px[trackID];
      const double y = py[trackID];
      const double z = pz[trackID];
      const double pt = std::hypot(x, y);
      const double p = std::sqrt(x * x + y * y + z * z);
      const double eta = safeEta(p, z);
      const double phi = std::atan2(y, x);
      const int pdgCode = pdg[trackID];
      const double startX = vx[trackID];
      const double startY = vy[trackID];
      const double startZ = vz[trackID];
      const double dx = startX - *eventVx;
      const double dy = startY - *eventVy;
      const double dz = startZ - *eventVz;
      const double radiusFromEventXY = std::hypot(dx, dy);
      const uint32_t prop = static_cast<uint32_t>(propRaw[trackID]);
      const bool keptByO2 = (prop & 0x1U) != 0;
      const int process = (prop >> 1) & 0x3fU;
      const uint32_t hitMask = (prop >> 7) & ((uint32_t{1} << 22) - 1);
      const bool inhibited = ((prop >> 30) & 0x1U) != 0;
      const bool toBeDone = ((prop >> 31) & 0x1U) != 0;
      const bool isTransported = toBeDone && !inhibited;
      const bool isPrimary = process == 0 || (mother[trackID] < 0 && secondMother[trackID] < 0);
      const bool hasDaughters = daughterCounts[trackID] > 0 || firstDaughter[trackID] >= 0 || lastDaughter[trackID] >= 0;
      const bool canAvoid = !keptByO2 && !hasDaughters;

      if (downsample) {
        const int phiBin = clippedBin(phi, -M_PI, M_PI, nPhiBins);
        const int radiusBin = clippedBin(radiusFromEventXY, 0.0, radiusMax, nRadiusBins);
        const int etaBin = clippedBin(eta, etaMin, etaMax, nEtaBins);
        const int labelBin = canAvoid ? 1 : 0;
        const int binIndex = balanceBinIndex(phiBin, radiusBin, etaBin, labelBin, nPhiBins, nRadiusBins, nEtaBins);
        if (binCounts[binIndex] >= quotaPerBin) {
          continue;
        }
        ++binCounts[binIndex];
      }

      const double mass = massFromPDG(pdgCode);
      const double energy = std::sqrt(std::max(0.0, mass * mass + p * p));

      out << sourceID << ',' << entry << ',' << trackID << ',' << mcLabelRaw(trackID, entry, sourceID) << ','
          << currentFile << ',' << pdgCode << ',' << std::abs(pdgCode) << ',' << chargeSign(pdgCode) << ','
          << mass << ',' << energy << ',' << energy - mass << ','
          << x << ',' << y << ',' << z << ',' << p << ',' << pt << ',' << eta << ','
          << phi << ',' << (p > 0 ? std::acos(z / p) : std::numeric_limits<double>::quiet_NaN()) << ','
          << safeRapidity(energy, z) << ','
          << startX << ',' << startY << ',' << startZ << ',' << tns[trackID] << ','
          << *eventVx << ',' << *eventVy << ',' << *eventVz << ','
          << dx << ',' << dy << ',' << dz << ','
          << std::hypot(startX, startY) << ',' << radiusFromEventXY << ',' << std::sqrt(dx * dx + dy * dy + dz * dz) << ','
          << mother[trackID] << ',' << secondMother[trackID] << ',' << firstDaughter[trackID] << ',' << lastDaughter[trackID] << ','
          << daughterCounts[trackID] << ',' << hasDaughters << ','
          << process << ',' << statusCode[trackID] << ',' << weight[trackID] << ','
          << hitMask << ',' << popcount22(hitMask) << ',' << (hitMask != 0) << ','
          << toBeDone << ',' << inhibited << ',' << isTransported << ',' << isPrimary << ','
          << keptByO2 << ',' << canAvoid << '\n';

      ++totalRows;
      if (progressEvery > 0 && totalRows % progressEvery == 0) {
        std::cerr << "wrote " << totalRows << " rows; current file=" << currentFile << " event=" << entry << "\n";
      }
      if (maxTracks >= 0 && totalRows >= maxTracks) {
        std::cout << "Wrote " << totalRows << " rows to " << output << "\n";
        return;
      }
    }
    ++entry;
  }

  std::cout << "Wrote " << totalRows << " rows from " << files.size() << " file(s) to " << output << "\n";
}
