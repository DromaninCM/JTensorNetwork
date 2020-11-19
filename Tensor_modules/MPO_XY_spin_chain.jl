function MPO_XY(J)

S,Id = getLocalSpace("Spin",0.5);
zeros_2x2 = 0.0* Matrix(I, 2,2);

V1 = vcat(Id,S[:,1,:]);
V1 = vcat(V1,S[:,2,:]);
V1 = vcat(V1,zeros_2x2);

V2 = vcat(zeros_2x2,zeros_2x2);
V2 = vcat(V2,zeros_2x2);
V2 = vcat(V2,J*S[:,1,:]');
Hloc = cat(V1,V2;dims=2);

V2 = vcat(zeros_2x2,zeros_2x2);
V2 = vcat(V2,zeros_2x2);
V2 = vcat(V2,J*S[:,2,:]');
Hloc = cat(Hloc,V2;dims=2);

V2 = vcat(zeros_2x2,zeros_2x2);
V2 = vcat(V2,zeros_2x2);
V2 = vcat(V2,Id);
Hloc = cat(Hloc,V2;dims=2);

save = Hloc[:,:];
Hloc = [];

for i in 1:2:(Int(size(save,1)-1))
	if i == 1
		Hloc = vcat(save[1:2,i],save[1:2,i+1]);
		for j in 3:2:(Int(size(save,1)-1));
			Hloc = vcat(Hloc,save[j:j+1,i],save[j:j+1,i+1]);
		end
	else
		V2 = vcat(save[1:2,i],save[1:2,i+1]);
		for j in 3:2:(Int(size(save,1)-1));
			V2 = vcat(V2,save[j:j+1,i],save[j:j+1,i+1]);
		end
		Hloc = cat(Hloc,V2;dims=2);
	end
end

Hloc = reshape(Hloc,(2,2,4,4));
Hloc = permutedims(Hloc,[3 1 4 2]);

return Hloc;

end
