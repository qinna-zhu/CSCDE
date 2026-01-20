% ============ Main code for CSCDE  ================
clc;clear all;
rng('default');
warning('off');
sn1=1;
maxfes=1000;
funcnum_set=[1,2,3,4,5,10,16,19];
for kkkk=7:7
    % ----------definition of problems-------------------
    funcnum =funcnum_set(kkkk); % funcnum chosen from {1,2,3,4,5,10,16,19}
    % ---------------------------------------------------
    if funcnum==1 %---Ellipsoid
        fname='ellipsoid';
        Xmin=-5.12;Xmax=5.12;
        VRmin=-5.12;VRmax=5.12;
    elseif funcnum==2 %---Rosenbrock
        fname='rosenbrock';
        Xmin=-2.048;Xmax=2.048;
        VRmin=-2.048;VRmax=2.048;
    elseif funcnum==3 %---Ackley
        fname='ackley';
        Xmin=-32.768;Xmax=32.768;
        VRmin=-32.768;VRmax=32.768;
    elseif funcnum==4 %---Griewank
        fname='griewank';
        Xmin=-600;Xmax=600;
        VRmin=-600;VRmax=600;
    elseif funcnum==5 %---Rastrigins
        fname='rastrigin';
        Xmin=-5.12;Xmax=5.12;
        VRmin=-5.12;VRmax=5.12;
    elseif funcnum==16 || funcnum==19 % CEC 2005 function F10/F16/F19
        fname='benchmark_func';
        Xmin=-5;Xmax=5;
        VRmin=-5;VRmax=5;
    elseif funcnum==10
        fname='benchmark_func1';
        Xmin=-5;Xmax=5;
        VRmin=-5;VRmax=5;
    end
    pro_dim =100; % problem dimension
    popsize = 50; % population size
    LHSSize =2*popsize;
    F_weight_org =0.5; % F_weight
    F_CR_org = 0.8; % crossover probabililty
    %---------------------------------------------------------
    lu = [Xmin.* ones(1, pro_dim); Xmax.* ones(1, pro_dim)]; % problem domain
    FVr_minbound = lu(1,:); % Common Benchmark Function
    FVr_maxbound = lu(2,:); % Common Benchmark Function
    I_bnd_constr = 1;  %1: use bounds as bound constraints, 0: no bound constraints
    %--------

    runnum = 20; % number of runs
    for run = 1:runnum
        gfs=zeros(1,fix(maxfes/sn1));  CE=zeros(maxfes,2);
        fes = 0;
        flag_num = 0; % update delay
        LHS_p = zeros(LHSSize,pro_dim);
        LHS_f = zeros(LHSSize,1);
        position = zeros(popsize, pro_dim);
        fitness = zeros(popsize,1);
        % ---------Database initialization -------------------
        XRRmin = repmat(lu(1, :), LHSSize, 1);
        XRRmax = repmat(lu(2, :), LHSSize, 1);
        LHS_p = XRRmin + (XRRmax - XRRmin) .* lhsdesign(LHSSize, pro_dim);
        % fitness calculation
        for i=1:LHSSize
            LHS_f(i)= feval(fname, LHS_p(i,:)'); % for basic function
            fes = fes +1;
            CE(i,:)=[i,LHS_f(i)]; 
            if mod (i,sn1)==0
                cs1=i/sn1;
                gfs(1,cs1)=min(CE(1:i,2));
            end
        end
        [bestever,Index] = min(LHS_f);
        bestsample = LHS_p(Index,:);
        DB = [LHS_p,LHS_f];
        % ---------Population initialization ------------------
        [LHS_f,Index] = sort(LHS_f);
        LHS_p = LHS_p(Index,:);
        position = LHS_p(1:popsize,:);
        fitness = LHS_f(1:popsize);
        [fitnessForOrg,pop_index] = sort(fitness);
        positionForOrg = position(pop_index,:);
        % -----------------------------------------------------
        stage=0;
        G=0;
        while(fes < maxfes)
            G= G+1;
            stage=stage+1;
            [fitnessForOrg,pop_index] = sort(fitness);
            positionForOrg = position(pop_index,:);
            %% Original space DE Learning
            temp_porg = zeros(2*popsize,pro_dim);
            temp_forg = zeros(2*popsize,1);
            for i = 1:popsize
                for jjj=1:2
                    x = positionForOrg(i,:);
                    % --------轮盘赌选择最优基向量辅助变异操作----------
                    p = 0.1;
                    nbest =ceil(p*popsize);
                    % 选择当前种群nbest个样本进行轮盘赌操作
                    DBbestFit = fitnessForOrg(1:nbest,end);
                    DBbestPos = positionForOrg(1:nbest,1:pro_dim);
                    DBbestFitmax = sum(DBbestFit)./DBbestFit;
                    DBbestSelPro = DBbestFitmax./sum(DBbestFitmax);
                    bestProSel = cumsum(DBbestSelPro,1);
                    r = rand;
                    rid = find(r<=bestProSel);
                    if isempty(rid)
                        rn = 1;
                    else
                        rn = rid(1);
                    end
                    gbestsample=DBbestPos(rn,1:pro_dim);
                    % Mutation
                    A = randperm(popsize);
                    A(A == i) = [];
                    a = A(1); b=A(2);
                    if jjj==1
                        current_solution= x;
                        opposite_solution = FVr_minbound+ FVr_maxbound - current_solution;
                        new_lower_bound = min(current_solution, opposite_solution);
                        new_upper_bound = max(current_solution, opposite_solution);
                        random_solution = new_lower_bound + rand(size(current_solution)) .* (new_upper_bound - new_lower_bound);
                        x= random_solution;
                    end
                    y = x+F_weight_org.*(gbestsample-x)+F_weight_org.*(positionForOrg(a,:)-positionForOrg(b,:));
                    % Crossover
                    z = zeros(size(x));
                    j0 = randi([1 numel(x)]);
                    for j = 1:numel(x)
                        if j == j0 || rand <= F_CR_org
                            z(j) = y(j);
                        else
                            z(j) =x(j);
                        end
                    end
                    if (I_bnd_constr == 1)
                        for j=1:pro_dim %----boundary constraints via bounce back-------
                            if (z(j) > FVr_maxbound(j))
                                z(j) = FVr_maxbound(j) - rand*(FVr_maxbound(j) - FVr_minbound(j));
                            end
                            if (z(j) < FVr_minbound(j))
                                z(j) = FVr_minbound(j) + rand*(FVr_maxbound(j) - FVr_minbound(j));
                            end
                        end
                    end
                    temp_porg((i-1)*2+jjj,:) = z;
                end

            end
            % Fitnesss Evaluation (RBF model assisted estimation)
            % construct a RBF model using neighbors of current trial population(Original space)
            NS=2*(pro_dim+1);hx = DB(:,1:pro_dim);hf = DB(:,end);
            phdis=real(sqrt(temp_porg.^2*ones(size(hx'))+ones(size(temp_porg))*(hx').^2-2*temp_porg*(hx')));
            [~,sidx]=sort(phdis,2);                       
            nidx=sidx; nidx(:,NS+1:end)=[];               
            nid=unique(nidx);
            trainx_org=hx(nid,:);   trainf_org=hf(nid);
            flag='cubic';
            [lambda, gamma]=RBF(trainx_org,trainf_org,flag);
            RBFOrg=@(x) RBF_eval(x,trainx_org,lambda,gamma,flag); % Orginal RBF
            temp_appfit_org = RBFOrg(temp_porg); 

            reduced_dim = 5;
            X = DB(:,1:pro_dim);  X = X';  Y = DB(:,end);  % select training samples for autoencoder training
            positionForSub = temp_porg;  % Autoencoder training and new sample encode
            c=temp_porg; % Decode the mutant vectorts to the Orginal space
            % Autoencoder training and new sample encode
            hiddenSize = reduced_dim;
            autoenc = trainAutoencoder(X,hiddenSize,'MaxEpochs',100,'ShowProgressWindow',false); % 使用数据库所有样本训练自编码器模型
            encodedX = encode(autoenc,X);   encodedX = encodedX'; % 编码源空间数据库样本得到数据库子空间投影样本
            encodedp = encode(autoenc,positionForSub');   encodedp = encodedp'; %编码源空间种群样本得到子空间投影种群
            encodeC=encode(autoenc,c') ; encodeC=encodeC';
            % % training samples for RBF modeling--neighborhood samples
            NS=2*(reduced_dim+1);
            phdis=real(sqrt(encodeC.^2*ones(size(encodedX'))+ones(size(encodeC))*(encodedX').^2-2*encodeC*(encodedX')));
            [~,sidx]=sort(phdis,2);                        % 每行都进行排序
            nidx=sidx; nidx(:,NS+1:end)=[];                % 个体的近邻样本指标集矩阵
            nid=unique(nidx);
            trainx_sub=encodedX(nid,:);   trainf_sub=Y(nid);
            flag='cubic';
            [lambda, gamma]=RBF(trainx_sub,trainf_sub,flag);
            RBFSub=@(x) RBF_eval(x,trainx_sub,lambda,gamma,flag); % 构造子空间RBF模型
            % ---------------------------------------------------------

            % Fitness estimation in subspace for sorting the swarm in subspace
            temp_appfit_sub = RBFSub(encodedp);  % 估计子空间投影种群适应度
            [temp_appfit_org_sort,id_org] = sort(temp_appfit_org);
            [temp_appfit_sub_sort,id_sub] = sort(temp_appfit_sub);
            infillsampleidx11 = [id_org(1); id_sub(1)];
            infillsampleidx=unique(infillsampleidx11);

            ns = length(infillsampleidx);
            p1 = zeros(ns,pro_dim);p2 = zeros(ns,pro_dim);
            f1 = zeros(ns,1);f2 = zeros(ns,1);
            i=1;
            while i<=ns
                tp_sub = encodedp(infillsampleidx(i),:); %获取编码子空间对应填充样本
                dectp_sub = decode(autoenc,tp_sub');    dectp_sub = dectp_sub';   % 解码对应填充样本
                tp_org = temp_porg(infillsampleidx(i),:); % 源空间采样指标对应样本

                p1(i,:) = dectp_sub;        % 子空间采集样本解码后的源空间样本
                p2(i,:) = tp_org;           % 源空间采集样本

                %% fmincon local search
                FE=1000;
                options = optimset('Algorithm','interior-point','Display','off','MaxFunEvals',FE,'TolFun',1e-8,'GradObj','off'); % run interior-point algorithm
                GP = (maxfes - fes + 1) / maxfes; AT = 0.5 * GP;
                if   (AT>rand || flag_num < 1)
                    iffes=0;
                    flag_num=1; %subspace update flag   
                    tfapporg = RBFOrg(dectp_sub);           
                    if tfapporg < bestever
                        tf_sub = feval(fname,dectp_sub'); 
                        iffes=1;
                        DB = [DB;dectp_sub,tf_sub];
                        f1(i) = tf_sub;
                        fes = fes+1;
                        if fes <= maxfes
                            CE(fes,:)=[fes,tf_sub];
                            if mod (fes,sn1)==0
                                cs1=fes/sn1;
                                gfs(1,cs1)=min(CE(1:fes,2));
                            end
                        end
                        % Selection
                        if tf_sub < bestever
                            stage=0;
                            bestsample = dectp_sub; bestever = tf_sub;flag_num=0;
                        end
                    end
                    if iffes==0
                        initialpoint_sub = tp_sub; % 局部搜索初始点
                        L_sub = min(encodedp);    U_sub=max(encodedp); % 围绕编码空间种群领域进行局部搜索
                        if isnan(RBFSub(initialpoint_sub))==0 
                            opt_sub = fmincon(RBFSub,initialpoint_sub,[],[],[],[],L_sub,U_sub,[],options);
                            % 将子空间SQP局部最优解码回源空间进行真实评价
                            decopt_sub = decode(autoenc,opt_sub');  decopt_sub = decopt_sub';   % 解码局部最优解到源空间
                            tfdecopt_sub = RBFOrg(decopt_sub); % 利用源空间RBF模型估计子空间解码后的填充样本适应度   
                            find_temp=RBFSub(opt_sub);
                            if  tfdecopt_sub<tfapporg
                                fit_sub = feval(fname,decopt_sub'); 
                                DB = [DB;decopt_sub,fit_sub];
                                fes = fes+1;
                                if fes <= maxfes
                                    CE(fes,:)=[fes,fit_sub];
                                    if mod (fes,sn1)==0
                                        cs1=fes/sn1;gfs(1,cs1)=min(CE(1:fes,2));
                                    end
                                end
                                % Selection
                                if fit_sub < bestever
                                    stage=0;
                                    bestsample = decopt_sub; bestever = fit_sub;flag_num=0;
                                end
                            end
                        end
                    end
                else
                    iffes=0;
                    tfapporg = RBFOrg(tp_org);
                    if tfapporg < bestever 
                        tf_org = feval(fname,tp_org');
                        DB = [DB;tp_org,tf_org];
                        iffes=1;
                        f2(i) = tf_org;
                        fes = fes+1;
                        if fes <= maxfes
                            CE(fes,:)=[fes,tf_org];
                            if mod (fes,sn1)==0
                                cs1=fes/sn1;
                                gfs(1,cs1)=min(CE(1:fes,2));
                            end
                        end
                        if tf_org < bestever
                            stage=0;
                            bestsample = tp_org; bestever = tf_org;
                        end
                    end 
                    if iffes==0
                        initialpoint_org = tp_org;
                        L_org = min(temp_porg);    U_org=max(temp_porg);
                        if isnan(RBFOrg(initialpoint_org)) == 0 % 判断是否未NAN
                            opt_org = fmincon(RBFOrg,initialpoint_org,[],[],[],[],L_org,U_org,[],options);
                            tfopt_org = RBFOrg(opt_org); 
                            if tfopt_org<tfapporg
                                fit_org = feval(fname,opt_org');  
                                DB = [DB;opt_org,fit_org];
                                fes = fes+1;
                                if fes <= maxfes
                                    CE(fes,:)=[fes,fit_org];
                                    if mod (fes,sn1)==0
                                        cs1=fes/sn1;
                                        gfs(1,cs1)=min(CE(1:fes,2));
                                    end
                                end
                                if fit_org < bestever
                                    stage=0;
                                    bestsample = opt_org;
                                    bestever = fit_org;
                                end
                            end
                        end
                    end
                end
                i=i+1;
            end
            %% 选择数据库中最优的样本作为迭代种群
            DB = unique(DB,'rows');
            DB = sortrows(DB,pro_dim+1);
            DB_size=min(size(DB,1),400);
            DB(DB_size+1:end,:)=[];
            position = DB(1:popsize,1:pro_dim);
            fitness = DB(1:1:popsize,pro_dim+1);
            %----------------------------------------------
            fitnessbestn = bestever; % output current global best.
            fprintf('OS_num_RunNo.: %d, CSCDE on %s, FunctionNo.: %d, evaluation: %d, best: %0.4g\n',run,fname,funcnum,fes,bestever)
        end
        gsamp1(run,:)=gfs;
    end

    best_samp=min(gsamp1(:,end));
    worst_samp=max(gsamp1(:,end));
    samp_mean=mean(gsamp1(:,end));
    samp_median=median(gsamp1(:,end));
    std_samp=std(gsamp1(:,end));
    out1=[best_samp,worst_samp,samp_median,samp_mean,std_samp];
    % 绘制收敛曲线
    gsamp1_ave=mean(gsamp1,1);
    gsamp1_log=log(gsamp1_ave);
    for j=1:maxfes
        if mod(j,sn1)==0
            j1=j/sn1; gener_samp1(j1)=j;
        end
    end

    figure(2);
    plot(gener_samp1,gsamp1_log,'.-k','Markersize',16)
    legend('CSCDE');
    xlabel('Function Evaluation Calls');
    ylabel('Mean Fitness Value (Natural Log)');
    set(gca,'FontSize',20);

    filename='.\results\'
    b1 = strcat("F",funcnum+"_",pro_dim+"D",".mat");
    c1=strcat(filename,b1);
    save(c1)
end

