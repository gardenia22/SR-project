clear al;clc;
cd ./s1_noise

noise_str = 'n005_';

% [wav,fs]=audioread('bbaf2n.wav');    
% size(wav)
% noisewav(1:size(wav))=0;
% for i=1:size(wav)
%     noisewav(i)=rand*(-1)^randi(1:2);    
% end
% wav2 = (wav(:)*1 + noisewav(:)*0.005);   
% audiowrite('bbaf2n_n0.wav',wav2,25000)
% 
% player=audioplayer(wav2, 25000);
% player.play  



dirlist = dir('*.wav');
length(dirlist)
for i = 1:length(dirlist)
    filename =  dirlist(i).name
    [wav,fs]=audioread(filename);    
    size(wav)
    noisewav = 0;
    noisewav(1:size(wav))=0;

    for j=1:size(wav)
        noisewav(j)=rand*(-1)^randi(1:2);    
    end
    
    wav2 = (wav(:)*1 + noisewav(:)*0.005);
    
    name2 = strcat(noise_str, filename);
    audiowrite(name2,wav2,25000)
     
end