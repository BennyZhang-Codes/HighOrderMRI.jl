mutable struct RebuildHOOpTestResource
    closed :: Bool
end

Base.close(resource::RebuildHOOpTestResource) = (resource.closed = true)


@testset "@rebuild_HOOp lifecycle" begin
    resource = RebuildHOOpTestResource(false)
    old_resource = resource
    constructor_saw_nothing = Ref(false)

    result = @rebuild_HOOp resource begin
        constructor_saw_nothing[] = resource === nothing
        RebuildHOOpTestResource(false)
    end

    @test old_resource.closed
    @test constructor_saw_nothing[]
    @test result === resource
    @test !resource.closed

    failed_resource = RebuildHOOpTestResource(false)
    old_failed_resource = failed_resource
    @test_throws ErrorException @rebuild_HOOp failed_resource error("constructor failed")
    @test old_failed_resource.closed
    @test failed_resource === nothing

    nothing_resource = nothing
    @rebuild_HOOp nothing_resource RebuildHOOpTestResource(false)
    @test nothing_resource isa RebuildHOOpTestResource
end
