function binwrite(filename,P)
%存储为二进制单精度文件
fid=fopen(filename,'wb');
fwrite(fid,P,'single');
fclose(fid);