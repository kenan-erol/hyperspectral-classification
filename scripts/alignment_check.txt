cd drop-4/'Ecstasy 2025-01-14'/Group/M0001/
info = enviinfo('measurement.hdr')

hcube = hypercube(info.Filename)

data = multibandread('measurement.raw',[info.Height, info.Width, info.Bands],...
info.DataType, info.HeaderOffset, info.Interleave, info.ByteOrder);

hyperspectralViewer(hcube)